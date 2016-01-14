% Max Jaderberg 28/3/14
% Convert cuda-convnet to mconvnet

function net = cudaconvnet_to_mconvnet(cnet_fn)

    if ~exist('cnet_fn')
        cnet_fn = 'models/charnet_layers.mat';
    end
    
    layers = load_nostruct(cnet_fn);

    % create a net
    net = neuralnetwork();
    
    % compose from cuda-convnet layers
    fields = fieldnames(layers);
    allowed_types = {'conv', 'fc', 'softmax', 'eltmax'};
    while ~isempty(fields)
        layer = layers.(fields{1});
        if ~any(ismember(allowed_types, layer.type))
            % not an allowed type so continue
            fields(1) = [];
            continue
        end
        % check if input layers have been added already
        inputs = {};
        not_added = false;
        for i=1:size(layer.inputLayers,1)
            input = strtrim(layer.inputLayers(i,:));
            input_layer = get_layer(layers, input);
            if strcmp(input_layer.type, 'data')
                % input is a data layer
                inputs{end+1} = ['data.' input_layer.name];
                continue
            end
            try
                net.idx_from_name(net, input);
            catch
                not_added = true;
            end
            inputs{end+1} = input;
        end
        if not_added
            % one of the input layers hasn't been added so move this layer
            % to the end of processing (so the input layers can be added
            % first).
            fields{end+1} = fields{1};
            fields(1) = [];
            continue
        end
        
        switch layer.type
            case 'conv'
                net = net.add_layer(net, conv_layer(layer.name, get_filters(layer), layer.biases), inputs);
            case 'fc'
                net = net.add_layer(net, fc_layer(layer.name, layer.weights, layer.biases), inputs);
            case 'softmax'
                %net = net.add_layer(net, softmax_layer(layer.name), inputs);
            case 'eltmax'
                net = net.add_layer(net, eltmax_layer(layer.name), inputs);
        end
        
        % this layer is done
        fields(1) = [];
    end
    
end

function weights = get_filters(layer)
    % weights is rows x cols x chans x filters

    weights = layer.weights;
    weights = reshape(weights, [size(weights,2) size(weights,3)]);
    sz = layer.filterSize;
    chans = layer.filterChannels;
    weights = reshape(weights, [sz sz chans size(weights,2)]);
    weights = permute(weights, [2 1 3 4]);

end

function layer = get_layer(layers, layer_name)

    fields = fieldnames(layers);
    
    for i=1:length(fields)
        layer = layers.(fields{i});
        if strcmp(layer.name, layer_name)
            break
        end
    end

end