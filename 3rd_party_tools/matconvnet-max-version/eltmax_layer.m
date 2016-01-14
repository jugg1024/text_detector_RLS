% Max Jaderberg 28/3/14
% Eltmax layer

function layer = eltmax_layer(name)

    layer.type = 'eltmax';
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
    
    assert(length(X) > 1);
    for i=2:length(X)
        assert(isequal(size(X{i}), size(X{i-1})));
    end

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

end

function layer = fprop(layer, X)

    maxM = [];
    for i=1:length(X)
        maxM = cat(3, maxM, reshape(X{i}, [], size(X{i},4)));
    end
    [M, layer.I] = max(maxM, [], 3);

    layer.Xout = reshape(M, size(X{1}));

end

function layer = fprop_gpu(layer, X)
   
    layer = fprop(layer, X);

end

function layer = bprop(layer, X, dzdy)

    layer.dzdx = {};
    for i=1:length(X)
        ismax = layer.I == i;
        ismax = reshape(ismax, size(X{1}));
        layer.dzdx{i} = dzdy .* ismax;
    end

end

function layer = bprop_gpu(layer, X, dzdy)
    layer = bprop(layer, X, dzdy);
end

function layer = update(layer)

end