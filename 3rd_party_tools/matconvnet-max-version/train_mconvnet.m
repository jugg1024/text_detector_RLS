% Max Jaderberg 21/4/14
% Train a network

function net = train_mconvnet(net, params, traindp, testdp, save_fn, stopn)

    if ~exist('stopn', 'var')
        stopn = 99999999; % bloody ages
    end

    batchsz = params.batchsz;
    testbatchsz = params.testbatchsz;
    minibatchsz = params.minibatchsz;
    data_name = params.data_name;
    label_name = params.label_name;
    cost_names = params.cost_names;
    test_freq = params.test_freq;

    trainerrs = {};
    testerrs = {};
    testbatchn = [];
    for i=1:length(cost_names)
        trainerrs{i} = [];
        testerrs{i} = [];
    end
    
    fprintf('----------------------------------------------------------------------------------------------------------------------------------------\n');
    fprintf('----------------------------------------------------------------------------------------------------------------------------------------\n');
    fprintf('TRAINING\n');
    
    while true
        % get new training batch
        start = tic;
        [traindp, data, labels] = traindp.get_next_batch(traindp, batchsz);
        dp_time = toc(start);
        
        fprintf('%d. %d samples:\n', traindp.batchnum, size(data, 4));
        
        errs = {};
        costs = {};
        for i=1:length(cost_names)
            errs{i} = [];
            costs{i} = [];
        end
        
        start = tic;
        % process in minibatches
        curs = 1;
        while curs <= size(data, 4)
            nextcurs = min(curs + minibatchsz - 1, size(data, 4));
            
            net = net.forward(net, struct(data_name, data(:,:,:,curs:nextcurs), label_name, labels(:,:,:,curs:nextcurs)));
            net = net.backward(net, struct(data_name, data(:,:,:,curs:nextcurs), label_name, labels(:,:,:,curs:nextcurs)));
            net = net.update(net);
            
            for i=1:length(cost_names)
                costlayer = net.get_layer(net, cost_names{i});
                errs{i} = [errs{i} costlayer.accuracy];
                costs{i} = [costs{i} costlayer.Xout];
                %fprintf('\t%s %f\n', cost_names{i}, mean(costlayer.accuracy));
            end
            
            
            curs = nextcurs + 1;
        end
        t = toc(start);
        
        batcherr = cellfun(@mean, errs, 'UniformOutput', false);
        batchcost = cellfun(@mean, costs, 'UniformOutput', false);
        
        for i=1:length(cost_names)
            fprintf('\t%s %f %f (%f seconds)\n', cost_names{i}, batcherr{i}, abs(batchcost{i}), dp_time + t);
            trainerrs{i}(end+1) = batcherr{i};
        end
        
        if rem(traindp.batchnum, test_freq) == 0
            % do testing
            fprintf('----------------------------------------------------------------------------------------------------------------------------------------\n');
            fprintf('----------------------------------------------------------------------------------------------------------------------------------------\n');
            fprintf('TESTING\n');
            start = tic;
            [testdp, data, labels] = testdp.get_next_batch(testdp, testbatchsz);
            testdp_time = toc(start);
            
            fprintf('%d. %d samples:\n', testdp.batchnum, size(data, 4));
            
            % turn off dropout
            for i=1:length(net.layers)
                if strcmp(net.layers{i}.type, 'conv')
                    net.layers{i}.drop_off = true;
                end
            end
            
            start = tic;
            net = net.forward(net, struct(data_name, data, label_name, labels));
            t = toc(start);
            
            % turn on dropouts again
            for i=1:length(net.layers)
                if strcmp(net.layers{i}.type, 'conv')
                    net.layers{i}.drop_off = false;
                end
            end
            
            for i=1:length(cost_names)
                costlayer = net.get_layer(net, cost_names{i});
                testerr = mean(costlayer.accuracy);
                testcost = mean(costlayer.Xout);
                fprintf('\t%s %f %f (%f seconds)\n', cost_names{i}, testerr, abs(testcost), testdp_time + t);
                testerrs{i}(end+1) = testerr;
            end
            testbatchn(end+1) = length(trainerrs{1});
            
            fprintf('\n\tWeight increments:\n');
            for i=1:length(net.layers)
                if strcmp(net.layers{i}.type, 'conv')
                    w_inc = mean(abs(net.layers{i}.w_inc(:)));
                    b_inc = mean(abs(net.layers{i}.b_inc(:)));
                    if w_inc > 0 || b_inc > 0
                        fprintf('\t%s: %f %f\n', net.layers{i}.name, w_inc, b_inc);
                    end
                end
                if strcmp(net.layers{i}.type, 'sepconv')
                    w_inc = mean([abs(net.layers{i}.h_inc(:)); abs(net.layers{i}.v_inc(:))]);
                    if w_inc > 0 || b_inc > 0
                        fprintf('\t%s: %f\n', net.layers{i}.name, w_inc);
                    end
                end
            end
            
            figure(1); 
            maxerr = 0;
            for i=1:length(cost_names)
                maxerrnew = max([trainerrs{i} testerrs{i}]);
                maxerr = max(maxerrnew, maxerr);
                plot(1:length(trainerrs{i}), trainerrs{i}, '--', testbatchn, testerrs{i}, '-');
                hold on;
            end
            hold off;
            xlabel('Epoch #'); ylabel('Error'); 
            axis([0 length(trainerrs{1}) 0 maxerr]);
            pause(0.05);
            
            net.save(net, save_fn);
            fprintf('saved to %s\n', save_fn);
            fprintf('----------------------------------------------------------------------------------------------------------------------------------------\n');
            fprintf('----------------------------------------------------------------------------------------------------------------------------------------\n');
            fprintf('TRAINING\n');
        end
        
        if traindp.batchnum > stopn
            break
        end
        
    end

end