% Max Jaderberg 21/4/14
% Element wise sum layer
% Xout = sum(coeff(n)*X{n});

function layer = eltsum_layer(name, coeff)

    if ~exist('coeff', 'var')
        coeff = 1;
    end

    layer.type = 'eltsum';
    layer.name = name;
    layer.coeff = coeff;
    layer.forward = @forward;
    layer.backward = @backward;
    layer.Xout = [];
    layer.Xout_gpu = [];
    layer.use_gpu = @use_gpu;
    layer.gpu = 0;
    layer.update = @(x) (x);

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
    
    assert(length(X) >= 1);
    assert(length(X) == length(layer.coeff));
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

    sumX = 0;
    for i=1:length(X)
        sumX = sumX + layer.coeff(i) * X{i};
    end

    layer.Xout = sumX;

end

function layer = fprop_gpu(layer, X)
   
    layer = fprop(layer, X);

end

function layer = bprop(layer, X, dzdy)

    layer.dzdx = {};
    for i=1:length(X)
        layer.dzdx{i} = layer.coeff(i) * dzdy;
    end

end

function layer = bprop_gpu(layer, X, dzdy)
    layer = bprop(layer, X, dzdy);
end
