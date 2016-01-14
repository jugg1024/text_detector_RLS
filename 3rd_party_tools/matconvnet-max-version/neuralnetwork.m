% Max Jaderberg 19/12/13
% Neural Network
% Initialize net:
%   net = neuralnetwork();
% Add layers:
%   net = net.add_layer(net, conv_layer('conv1_64', [5,3,64,0.0001]), 'data.X');
%   net = net.add_layer(net, conv_layer('conv1_16', [5,3,16,0.0001]), 'data.X');
%   net = net.add_layer(net, maxpool_layer('pool1', [8,2]), {'conv1_64', 'conv1_16'});
%   net = net.add_layer(net, fc_layer('fc2', [2,0.0001]), 'pool1');
% Generate data:
%   datastruct.X = single(randn(24, 24, 3, 128));
% Process data:
%   net = net.forward(net, datastruct);
% View outputs:
%   net.Xout
% View layer weights:
%   net.layers{net.idx_from_name(net, 'fc2')}.w


function net = neuralnetwork(varargin)

    net.layers = {};
    net.add_layer = @add_layer;
    net.forward = @forward;
    net.backward = @backward;
    net.update = @update;
    net.use_gpu = @use_gpu;
    net.gpu = 0;
    net.idx_from_name = @idx_from_name;
    net.inputs = {};
    net.input_names = {};
    net.outputs = {};
    net.Xout = [];
    net.verbose = false;
    net.get_layer = @get_layer;
    net.compute_graph = [];
    net.update_graph = @update_graph;
    net.save = @save_net;
    
end

function net = use_gpu(net, device)
    
    for i=1:length(net.layers)
        net.layers{i} = net.layers{i}.use_gpu(net.layers{i}, device);
    end
    
    net.gpu = device;

end

function net = add_layer(net, layer, inputs)

    net.layers{end+1} = layer;
    layer_idx = length(net.layers);
    net.input_names{end+1} = inputs;
    net.outputs{end+1} = {};
    
    if ischar(inputs)
        if strfind(inputs, 'data.')
            % input is a data field
            r = regexp(inputs, '\.', 'split');
            net.inputs{layer_idx} = r(2);
        else
            inidx = idx_from_name(net, inputs);
            net.inputs{layer_idx} = {inidx};
            net.outputs{inidx}{end+1} = layer_idx;
        end
    elseif iscell(inputs)
        net.inputs{layer_idx} = {};
        for i=1:length(inputs)
            if strfind(inputs{i}, 'data.')
                % input is a data field
                r = regexp(inputs{i}, '\.', 'split');
                net.inputs{layer_idx}{end+1} = r{2};
            else
                inidx = idx_from_name(net, inputs{i});
                net.inputs{layer_idx}{end+1} = inidx;
                net.outputs{inidx}{end+1} = layer_idx;
            end
        end
    else
        error('network.add_layer requires string or cell array of strings of inputs');
    end

end

function net = update_graph(net)
    %% update the grpah of computation order
    if net.verbose
        fprintf('Computation order:\n');
    end

    net.compute_graph = [];
    
    % process all layers
    processed = false(length(net.layers), 1);
    left_to_process = 1:length(net.layers);
    while ~all(processed)
        % find a layer whos inputs are either all processed, or only
        % takes data inputs
        for i=1:length(left_to_process)
            % assume its a good layer unless proved otherwise
            is_good_layer = true;
            left_to_process_idx = i;
            layer_idx = left_to_process(i);
            inputs = net.inputs{layer_idx};
            for j=1:length(inputs)
                if ~ischar(inputs{j})
                    % this takes a layer as its input
                    if ~processed(inputs{j})
                        % which hasn't been processed
                        is_good_layer = false;
                    end
                end
            end
            if is_good_layer
                break
            end
        end
        
        net.compute_graph(end+1) = layer_idx;
        if net.verbose
            fprintf('\t%d. %s (inputs:', length(net.compute_graph), net.layers{layer_idx}.name);
            for i = 1:length(net.inputs{layer_idx})
                in = net.inputs{layer_idx}{i};
                if ischar(in)
                    fprintf(' data.%s', in);
                else
                    fprintf(' %s', net.layers{in}.name);
                end
            end
            fprintf(')\n');
        end
            
        processed(layer_idx) = true;
        left_to_process(left_to_process_idx) = [];
    end
    
    %% backprop graph
    % start with reversed forward graph
    net.backcompute_graph = fliplr(net.compute_graph);
    % label each layer with whether the grad is actually needed
    back_needed = [];
    for layer_idx=net.backcompute_graph
        has_weights = {'conv', 'sepconv'};
        layer = net.layers{layer_idx};
        if any(ismember(has_weights, layer.type))
            if abs(layer.w_learning_rate) > 0 || abs(layer.b_learning_rate) > 0
                back_needed(end+1) = true;
                continue
            end
        end
        back_needed(end+1) = false;
    end
    stop = length(back_needed);
    for i=0:length(back_needed)
        if ~any(back_needed(i+1:end))
            % non of the remaining need grad so stop here
            stop = i;
            break
        end
    end
    net.backcompute_graph = net.backcompute_graph(1:stop);
    if net.verbose
        fprintf('Backprop for:\n');
        for i=1:length(net.backcompute_graph)
            layer_idx = net.backcompute_graph(i);
            fprintf('\t%d. %s\n', i, net.layers{layer_idx}.name);
        end
    end
end

function net = forward(net, data)

    if isempty(net.compute_graph)
        % compute processing order graph
        net = net.update_graph(net);
    end
    
    net.time = [];

    % process all layers
    for layer_idx=net.compute_graph
        
        inputs = net.inputs{layer_idx};
        
        % multiple inputs
        Xin = {};
        for i=1:length(inputs)
            if ischar(inputs{i})
                % this is a data input
                X = data.(inputs{i});
                if net.gpu > 0
                    X = gpuArray(single(X));
                end
            else
                if net.gpu > 0
                    X = net.layers{inputs{i}}.Xout_gpu;
                else
                    X = net.layers{inputs{i}}.Xout;
                end
            end
            Xin{end+1} = X;
        end
        
        % process the input data
        if net.verbose
            fprintf('%s (%s)\n', net.layers{layer_idx}.name, net.layers{layer_idx}.type);
            for i=1:length(Xin)
                fprintf('\tinput %d: %dx%dx%d\n', size(Xin{i},1), size(Xin{i},2), size(Xin{i},3));
            end
        end
        
        start = tic;
        net.layers{layer_idx} = net.layers{layer_idx}.forward(net.layers{layer_idx}, Xin);
        net.time(layer_idx) = toc(start);
        
        if net.verbose
            fprintf('\t--> %dx%dx%d\n', size(net.layers{layer_idx}.Xout,1), size(net.layers{layer_idx}.Xout,2), size(net.layers{layer_idx}.Xout,3));
        end
        
    end
    
    if net.gpu > 0
        net.Xout_gpu = net.layers{layer_idx}.Xout_gpu;
    else
        net.Xout = net.layers{layer_idx}.Xout;
    end

end

function net = backward(net, data)
    %% assumes that net.forward has been run before
    
    if isempty(net.compute_graph)
        % compute processing order graph
        net = net.update_graph(net);
    end
    try
        net.backcompute_graph;
    catch
        net = net.update_graph(net);
    end
    
    net.time = [];
    
    % process all layers (back to front)
    for layer_idx=net.backcompute_graph
        
        inputs = net.inputs{layer_idx};
        outputs = net.outputs{layer_idx};
        
        % accumulate multiple outputs
        dzdy = 0;
        for i=1:length(outputs)
            if ischar(outputs{i})
                % data layer who cares, no grad from that
                continue
            end
            % find which input is this layer
            for j=1:length(net.inputs{outputs{i}})
                if net.inputs{outputs{i}}{j} == layer_idx
                    break
                end
            end
            dzdy = dzdy + net.layers{outputs{i}}.dzdx{j};
        end
        
        % concatenate multiple inputs
        Xin = {};
        for i=1:length(inputs)
            if ischar(inputs{i})
                % this is a data input
                X = data.(inputs{i});
                if net.gpu > 0
                    X = gpuArray(single(X));
                end
            else
                if net.gpu > 0
                    X = net.layers{inputs{i}}.Xout_gpu;
                else
                    X = net.layers{inputs{i}}.Xout;
                end
            end
            Xin{end+1} = X;
        end

        if net.verbose
            fprintf('back %s (%s)\n', net.layers{layer_idx}.name, net.layers{layer_idx}.type);
        end
        
        start = tic;
        net.layers{layer_idx} = net.layers{layer_idx}.backward(net.layers{layer_idx}, Xin, dzdy);
        net.time(layer_idx) = toc(start);
        
    end


end

function net = update(net)

    try
        net.backcompute_graph;
    catch
        net = net.update_graph(net);
    end
    
    for i=net.backcompute_graph
        net.layers{i} = net.layers{i}.update(net.layers{i});
    end

end

function idx = idx_from_name(net, name)
    for i=1:length(net.layers)
        if strcmp(net.layers{i}.name, name)
            idx = i;
            return
        end
    end
    error('network layer %s does not exist', name);
end

function layer = get_layer(net, name)
    layer = net.layers{net.idx_from_name(net, name)};
end

function save_net(net, save_fn)

    % save space by removing all activiations
    net.Xout = [];
    for i=1:length(net.layers)
        net.layers{i}.Xout = [];
        net.layers{i}.dzdx = {};
        net.layers{i}.dzdw = [];
    end
    save(save_fn, 'net');

end