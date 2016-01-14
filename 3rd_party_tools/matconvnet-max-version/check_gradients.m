% Max Jaderberg 20/4/14
% Check gradients for layers

function check_gradients()
    close all;

    X = randn([24, 24, 3], 'single');

    %% Conv layer
    layer = conv_layer('conv', [9 3 8 0.1]);
    gradcheck(layer, X, 2.5);
    
    %% Softmax layer
    layer = softmax_layer('softmax');
    gradcheck(layer, X, 0.01);
    
    %% Check softmaxlogregcost is similar to softmax + logregcost
    X = randn(1, 1, 48, 'single');
    gt = rand(1, 1, 48) > 0.5;
    
    % individual layers
    smlayer = softmax_layer('softmax');
    smlayer = smlayer.forward(smlayer, {X});
    lrlayer = logregcost_layer('logreg_cost');
    lrlayer = lrlayer.forward(lrlayer, {X, gt});
    lrlayer = lrlayer.backward(lrlayer, {X, gt});
    smlayer = smlayer.backward(smlayer, {X}, lrlayer.dzdx);
    dzdx = smlayer.dzdx{1};
    y = lrlayer.Xout;
    
    % joint layer
    layer = softmaxlogregcost_layer('smlr');
    layer = layer.forward(layer, {X, gt});
    layer = layer.backward(layer, {X, gt}, []);
    dzdx_ = layer.dzdx{1};
    y_ = layer.Xout;
    
    figure() ; clf ;
    hist(abs(y(:) - y_(:))) ;
    figure() ; clf ;
    hist(abs(dzdx(:) - dzdx_(:))) ;
    
end

function gradcheck(layer, X, delta)
    
    % from layer
    layer = layer.forward(layer, {X});
    y = layer.Xout;
    dzdy = randn(size(layer.Xout),'single') ;
    layer = layer.backward(layer, {X}, dzdy);
    dzdx = layer.dzdx{1};
    
    % numerically
    dzdx_=zeros(size(dzdx));
    for i=1:numel(X)
      dx = zeros(size(X)) ;
      dx(i) = delta ;
      layer = layer.forward(layer, {X + dx});
      y_=layer.Xout;
      dzdx_(i) = dzdx_(i) + sum(sum(sum(sum(dzdy .* (y_ - y)/delta)))) ;
    end
    
    figure() ; clf ;
    hist(abs(dzdx(:) - dzdx_(:))) ;
    
end
