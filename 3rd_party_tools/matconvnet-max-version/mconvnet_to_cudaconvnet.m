% Max Jaderberg 3/4/14
% Save filters from an mconvnet for use in cuda-convnet

function mconvnet_to_cudaconvnet(net, save_dir)

    if ~exist('net', 'var')
        net = cudaconvnet_to_mconvnet;
    end
    if ~exist('save_dir', 'var')
        save_dir = '/Users/jaderberg/Data/TextSpotting/results/savedfilters';
    end

    vl_xmkdir(save_dir);
    
    for i=1:length(net.layers)
        layer = net.layers{i};
        if strcmp(layer.type, 'conv')
            w = layer.w;
            w = permute(w, [2 1 3 4]);
            w = reshape(w, [], size(w,4));
            b = layer.b;
            b = reshape(b, [], 1);
        elseif strcmp(layer.type, 'fc')
            w = layer.w;
            w = reshape(w, size(w,1), []);
            b = layer.b;
            b = reshape(b, [], 1);
        else
            continue
        end
        fnw = fullfile(save_dir, [layer.name '_w.mat']);
        save(fnw, 'w');
        fnb = fullfile(save_dir, [layer.name '_b.mat']);
        save(fnb, 'b');
        
        fprintf('Saved to %s and %s\n', fnw, fnb);
    end

end