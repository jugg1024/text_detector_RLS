% Max Jaderberg 21/4/14
% Sum Squares cost layer

function layer = sumsquarescost_layer(name, varargin)
    
    layer.type = 'sumsquarescost';
    layer.name = name;
    layer.forward = @forward;
    layer.backward = @backward;
    layer.Xout = [];
    layer.Xout_gpu = [];
    layer.accuracy = [];
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

    if layer.gpu
        layer = fprop_gpu(layer, X);
    else
        layer = fprop(layer, X);
    end

end

function layer = backward(layer, X, dzdy)

    if layer.gpu
        layer = bprop_gpu(layer, X, dzdy);
    else
        layer = bprop(layer, X, dzdy);
    end
    
    layer.dzdx = {layer.dzdx};

end

function layer = fprop(layer, X)

    x = reshape(X{1}, [], size(X{1},4));
    y = sum(x.^2,1);
    layer.accuracy = y;
    layer.Xout = y;

end

function layer = fprop_gpu(layer, X)

    layer = fprop(layer, X);

end

function layer = bprop(layer, X, dzdy)

    layer.dzdx = -2*X{1};

end

function layer = bprop_gpu(layer, X, dzdy)
    layer = bprop(layer, X, dzdy);
end

function layer = update(layer)

end