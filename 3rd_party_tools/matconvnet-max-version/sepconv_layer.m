% Max Jaderberg 27/4/14
% Separable Basis Convolution Layer
% inputs is either (k x k x n_filters)
% or [k, n_filters, initw] to intialize weights

function layer = sepconv_layer(name, varargin)

    if nargin == 0
        error('%s requires input params w or [k, n_filters, initw]', name);
    end

    if ndims(varargin{1}) == 2 && numel(varargin{1}) == 3
        % need to initialize w ourselves
        params = varargin{1};
        w = zeros(params(1), params(1), params(2));
        for i=1:params(2)
            w(:,:,i) = params(3)*randn(params(1),1)*randn(1,params(1));
        end
    else
        w = varargin{1};
    end
    
    % separate the filters
    v = zeros(size(w,1), 1, size(w,3));
    h = zeros(1, size(w,2), size(w,3));
    for i=1:size(w,3)
        assert(rank(w(:,:,i)) == 1);
        [U,S,V] = svd(w(:,:,i));
        sq = sqrt(S(1,1));
        v(:,:,i) = U(:,1) * sq;
        h(:,:,i) = V(:,1)' * sq;
    end
    
    layer.type = 'sepconv';
    layer.name = name;
    layer.w = single(w);
    layer.h = single(h);
    layer.v = single(v);
    layer.w_gpu = [];
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
    layer.h_inc = single(zeros(size(h)));
    layer.v_inc = single(zeros(size(v)));
    layer.dropout = 0;
    layer.drop_off = false;
    layer.training_params = @training_params;
    layer.fake_h = [];
    layer.fake_v = [];
    
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

    assert(size(X,1) >= size(layer.w,1));
    assert(size(X,2) >= size(layer.w,2));
    
    %% REAL FPROP
    a = gsepconv(X, layer.h, layer.v);
    layer.Xout = a(:,:,:,1:end-1);
    
    %% FAKE FPROP
%     if isempty(layer.fake_h)
%         layer.fake_h = zeros(size(layer.h,1), size(layer.h,2), size(X,3), size(X,3)*size(layer.h,3), 'single');
%         layer.fake_v = zeros(size(layer.v,1), size(layer.v,2), size(X,3)*size(layer.h,3), size(X,3)*size(layer.h,3), 'single');
%         
%         for i=1:size(X,3)
%             for j=1:size(layer.h,3)
%                 idx = (i-1)*size(layer.h,3)+j;
%                 layer.fake_h(:,:,i,idx) = layer.h(:,:,j);
%                 layer.fake_v(:,:,idx,idx) = layer.v(:,:,j);
%             end
%         end
%     end
% 
%     Xtemp = gconv(X, layer.fake_h);
%     layer.Xout = gconv(Xtemp, layer.fake_v);
    
end

function layer = fprop_gpu(layer, X)

    layer = fprop(layer, X);

end

function layer = bprop(layer, X, dzdy)

    %% REAL BPROP (NOT IMPLEMENTED)
%     [layer.dzdh, layer.dzdv, layer.dzdx] = gsepconv(X, layer.h, layer.v, dzdy) ;

    %% FAKE BPROP
    if isempty(layer.fake_h)
        layer.fake_h = zeros(size(layer.h,1), size(layer.h,2), size(X,3), size(X,3)*size(layer.h,3), 'single');
        layer.fake_v = zeros(size(layer.v,1), size(layer.v,2), size(X,3)*size(layer.h,3), size(X,3)*size(layer.h,3), 'single');
        for i=1:size(X,3)
            for j=1:size(layer.h,3)
                idx = (i-1)*size(layer.h,3)+j;
                layer.fake_h(:,:,i,idx) = layer.h(:,:,j);
                layer.fake_v(:,:,idx,idx) = layer.v(:,:,j);
            end
        end
    end
    
    Xtemp = gconv(X, layer.fake_h);
    [dzdfakev, dzdtemp] = gconv(Xtemp, layer.fake_v, dzdy) ;
    [dzdfakeh, layer.dzdx] = gconv(X, layer.fake_h, dzdtemp);
    
    layer.dzdv = zeros(size(layer.v));
    layer.dzdh = zeros(size(layer.h));
    
    for i=1:size(X,3)
        for j=1:size(layer.h,3)
            idx = (i-1)*size(layer.h,3)+j;
            layer.dzdh(:,:,j) = layer.dzdh(:,:,j) + dzdfakeh(:,:,i,idx);
            layer.dzdv(:,:,j) = layer.dzdv(:,:,j) + dzdfakev(:,:,idx,idx);
        end
    end
    layer.dzdh = layer.dzdh / size(X,3);
    layer.dzdv = layer.dzdv / size(X,3);

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
    
    % same as cuda-convnet https://code.google.com/p/cuda-convnet/wiki/LayerParams
    layer.h_inc = layer.w_momentum * layer.h_inc...
                - layer.weight_decay * epsW * layer.h...
                + epsW * layer.dzdh;
    layer.v_inc = layer.w_momentum * layer.v_inc...
                - layer.weight_decay * epsW * layer.v...
                + epsW * layer.dzdv;
    
    layer.h = layer.h + layer.h_inc;
    layer.v = layer.v + layer.v_inc;
    
    
    % UPDATE FAKE
    layer.fake_h = layer.fake_h * 0;
    layer.fake_v = layer.fake_v * 0;
    for i=1:size(layer.fake_h,3)
        for j=1:size(layer.h,3)
            idx = (i-1)*size(layer.h,3)+j;
            layer.fake_h(:,:,i,idx) = layer.h(:,:,j);
            layer.fake_v(:,:,idx,idx) = layer.v(:,:,j);
        end
    end
    
end

function layer = training_params(layer, params)
    layer.w_learning_rate = params(1);
    layer.weight_decay = params(2);
    layer.w_momentum = params(3);
    layer.dropout = params(4);
    layer.drop_off = params(5);
    
    % reset inc
    layer.w_inc = single(zeros(size(layer.w)));
end