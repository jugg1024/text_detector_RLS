% Max Jaderberg 19/12/13
% Convolution Layer
% inputs is either (h x w x n_inchans x n_filters), (n_filters x 1) 
% or [k, chans, n_filters, initw] to intialize weights

function layer = conv_layer(name, varargin)

    if nargin == 0
        error('%s requires input params w,b or [h, w, chans, n_filters, initw]', name);
    end

    if ndims(varargin{1}) == 2 && numel(varargin{1}) == 5
        % need to initialize w ourselves
        params = varargin{1};
        w = params(5)*randn(params(1), params(2), params(3), params(4));
        b = params(5)*randn(params(4), 1);
    elseif ndims(varargin{1}) ~= 4 && ndims(varargin{2}) ~= 2
        error('%s expects input w dims (h x w x n_inchans x n_filters) or [h, w, chans, n_filters, initw]', name);
    else
        w = varargin{1};
        b = varargin{2};
    end
    
    layer.type = 'conv';
    layer.name = name;
    layer.w = single(w);
    layer.w_gpu = [];
    layer.b = single(b);
    layer.b_gpu = [];
    layer.forward = @forward;
    layer.backward = @backward;
    layer.Xout = [];
    layer.Xout_gpu = [];
    layer.use_gpu = @use_gpu;
    layer.gpu = 0;
    layer.update = @update;
    layer.w_learning_rate = 0;
    layer.b_learning_rate = 0;
    layer.weight_decay = 0; % 0.0005
    layer.w_momentum = 0;
    layer.b_momentum = 0;
    layer.w_inc = single(zeros(size(w)));
    layer.b_inc = single(zeros(size(b)));
    layer.dropout = 0;
    layer.drop_off = false;
    layer.training_params = @training_params;
    
end

function layer = use_gpu(layer, v)

    if v > 0
        layer.gpu = v;
        % set gpu device
        d = gpuDevice();
        if d.Index ~= v
            gpuDevice(v);
        end
        % copy model to gpu
        layer.w_gpu = gpuArray(single(layer.w));
        layer.b_gpu = gpuArray(single(layer.b));
        % copy output to gpu
        layer.Xout_gpu = gpuArray(single(layer.Xout));
    else
        if layer.gpu > 0
            layer.gpu = 0;
            % copy back to cpu
            layer.w = gather(layer.w_gpu);
            layer.b = gather(layer.b_gpu);
            layer.Xout = gather(layer.Xout_gpu);
        end
    end

end

function layer = forward(layer, X)

    assert(length(X) == 1);
    X = X{1};
    
    if layer.gpu
        layer = fprop_gpu(layer, X);
    else
        layer = fprop(layer, X);
    end
    
    if layer.dropout > 0
        if layer.drop_off
            % rescale
            layer.Xout = (1 - layer.dropout) * layer.Xout;
        else
            % apply dropout mask
            layer.dropout_mask = rand(size(layer.Xout)) > layer.dropout;
            layer.Xout = layer.dropout_mask .* layer.Xout;
        end
    end

end

function layer = backward(layer, X, dzdy)

    X = X{1};
    
    if layer.dropout > 0 && ~layer.drop_off
        dzdy = layer.dropout_mask .* dzdy;
    end
    
    if layer.gpu
        layer = bprop_gpu(layer, X, dzdy);
    else
        layer = bprop(layer, X, dzdy);
    end

    layer.dzdx = {layer.dzdx};
    
end

function layer = fprop(layer, X)

    assert(size(X,3) == size(layer.w, 3));
    assert(size(X,1) >= size(layer.w,1));
    assert(size(X,2) >= size(layer.w,2));
    
    layer.Xout = gconv(X, layer.w);
    layer.Xout = bsxfun(@plus, layer.Xout, permute(layer.b, [2 3 1]));
    
end

function layer = fprop_gpu(layer, X)

    layer = fprop(layer, X);

end

function layer = bprop(layer, X, dzdy)
    
    [layer.dzdw, layer.dzdx] = gconv(X, layer.w, dzdy) ;
    layer.dzdb = squeeze(sum(sum(sum(dzdy,4),2),1)) ;

end

function layer = bprop_gpu(layer, X, dzdy)
    layer = bprop(layer, X, dzdy);
end

function layer = update(layer)
    
    % scale learning rate by 1/batchsz
    try
        batchsz = size(layer.Xout,4);
    catch
        batchsz = 1;
    end
    epsW = layer.w_learning_rate / batchsz;
    epsB = layer.b_learning_rate / batchsz;
    
    % same as cuda-convnet https://code.google.com/p/cuda-convnet/wiki/LayerParams
    layer.w_inc = layer.w_momentum * layer.w_inc...
                - layer.weight_decay * epsW * layer.w...
                + epsW * layer.dzdw;
    layer.b_inc = layer.b_momentum * layer.b_inc...
                + epsB * layer.dzdb;
    
    layer.w = layer.w + layer.w_inc;
    layer.b = layer.b + layer.b_inc;    
    
end

function layer = training_params(layer, params)
    layer.w_learning_rate = params(1);
    layer.b_learning_rate = params(2);
    layer.weight_decay = params(3);
    layer.w_momentum = params(4);
    layer.b_momentum = params(5);
    layer.dropout = params(6);
    layer.drop_off = params(7);
    
    % reset inc
    layer.w_inc = single(zeros(size(layer.w)));
    layer.b_inc = single(zeros(size(layer.b)));
end