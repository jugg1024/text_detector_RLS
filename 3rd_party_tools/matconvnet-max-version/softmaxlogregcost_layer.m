% Max Jaderberg 17/4/13
% Combined softmax + logreg cost layer
% Better than separate layers for numerical stability

function layer = softmaxlogregcost_layer(name, varargin)
    
    layer.type = 'softmaxlogregcost';
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

    assert(length(X) == 2);
    assert(isequal(size(X{1}), size(X{2})));
    % this is classification loss
    assert(size(X{1},1) == 1 && size(X{1},2) == 1);

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

    fc = squeeze(X{1});
    gt = squeeze(X{2});
    
    [~,p] = max(fc);
    [~,c] = max(gt);
    layer.accuracy = ~(p == c);

    % this is doing softmax elementwise over channels
    eX = exp(fc);
    sumeX = log(sum(eX, 1));
    allout = bsxfun(@minus, fc, sumeX);
    layer.Xout = allout(logical(gt))';

end

function layer = fprop_gpu(layer, X)

    layer = fprop(layer, X);

end

function layer = bprop(layer, X, dzdy)
    
    fc = squeeze(X{1});
    gt = squeeze(X{2});
    eX = exp(fc);
    sumeX = sum(eX, 1);
    probs = bsxfun(@rdivide, eX, sumeX);
    
    layer.dzdx = reshape(gt - probs, size(X{1},1), size(X{1},2), size(probs,1), size(probs,2));

end

function layer = bprop_gpu(layer, X, dzdy)
    layer = bprop(layer, X, dzdy);
end

function layer = update(layer)

end