% Max Jaderberg 20/12/13
% Example feed-forward use of neuralnetwork

%% Load data
X = single(randn(24, 24, 3, 12800));
y = single(randi([0,1], 1, 12800));

%% New network
net = neuralnetwork();

%% Add layers
% conv->relu->pool->conv->relu->pool->fc->relu->softmax
net = net.add_layer(net, conv_layer('conv1', [5,3,64,0.0001]), 'data.X');
net = net.add_layer(net, relu_layer('relu1'), 'conv1');
net = net.add_layer(net, avgpool_layer('pool1', [3, 2]), 'relu1');
net = net.add_layer(net, conv_layer('conv2', [5,64,256,0.0001]), 'pool1');
net = net.add_layer(net, relu_layer('relu2'), 'conv2');
net = net.add_layer(net, avgpool_layer('pool2', [3, 2]), 'relu2');
net = net.add_layer(net, fc_layer('fc2', [2,0.0001]), 'pool2');
net = net.add_layer(net, relu_layer('relu3'), 'fc2');
net = net.add_layer(net, softmax_layer('probs'), 'relu3');

net = net.use_gpu(net, 2);

%% Feed data
batchsz = 128;
for i=1:batchsz:size(X,4)
    tic;
    data.X = X(:,:,:,i:i+batchsz-1);
    data.y = y(:,i:i+batchsz-1);
    net = net.forward(net, data);
    toc
end
