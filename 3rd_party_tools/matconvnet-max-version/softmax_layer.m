% Max Jaderberg 19/12/13
% Softmax layer
% This can work after fc layers or conv layers on feature maps

function layer = softmax_layer(name, varargin)
    
    layer.type = 'softmax';
    layer.name = name;
    layer.forward = @forward;
    layer.backward = @backward;
    layer.Xout = [];
    layer.Xout_gpu = [];
    layer.use_gpu = @use_gpu;
    layer.gpu = 0;
    layer.update = @update;
    
end

function layer = use_gpu(layer, v)

    if v > 0
        layer.gpu = v;
        % set gpu device
        d = gpuDevice();
        if d.Index ~= v
            gpuDevice(v);
        end
        % copy output to gpu
        layer.Xout_gpu = gpuArray(single(layer.Xout));
    else
        if layer.gpu > 0
            layer.gpu = 0;
            % copy back to cpu
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

end

function layer = backward(layer, X, dzdy)

    X = X{1};

    if layer.gpu
        layer = bprop_gpu(layer, X, dzdy);
    else
        layer = bprop(layer, X, dzdy);
    end

    layer.dzdx = {layer.dzdx};
    
end

function layer = fprop(layer, X)

    % this is doing softmax elementwise over channels
    eX = exp(X);
    sumeX = sum(eX, 3) .^ -1;
    layer.Xout = repmat(sumeX, [1, 1, size(X,3), 1]) .* eX;

end

function layer = fprop_gpu(layer, X)

    layer = fprop(layer, X);

end

function layer = bprop(layer, X, dzdy)

    if dzdy == 0
        layer.dzdx = 0;
        return
    end
    
    Y = layer.Xout;
    Y = reshape(permute(Y, [3 1 2 4]), size(Y,3), []);
    dzdy = reshape(permute(dzdy, [3 1 2 4]), size(dzdy,3), []);
    
    layer.dzdx = Y .* bsxfun(@minus, dzdy, sum(dzdy .* Y,1));
    layer.dzdx = reshape(layer.dzdx, size(layer.dzdx,1), size(X,1), size(X,2), size(X,4));
    layer.dzdx = permute(layer.dzdx, [2 3 1 4]);

end

function layer = bprop_gpu(layer, X, dzdy)
    layer = bprop(layer, X, dzdy);
end

function layer = update(layer)

end