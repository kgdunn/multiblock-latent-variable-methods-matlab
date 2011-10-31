% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% Licensed under the BSD license.
% -------------------------------------------------------------------------

% Load the raw data
data = load('datasets/SBR.mat');
Y_data = data.Y;
   
% Specify the data dimensions
nBatches = 53;
nSamples = 200;
nTags = 6;
tagNames = {'Reactor temperature', 'Cooling water temperature', ...
            'Reactor jacket temperature', 'Latex density', ...
            'Conversion', 'Energy released'}';
        
% We must create a batch block first: tell it how many 
% batches there are in the aligned data
% Ignore tags 1, 2, 3: they are noisy and uninformative

batchX = block(data.X(:, 4:9), 'X: batch data',...     % name of the block
                 {'batch_tag_names', tagNames}, ...    % trajectory names
                 {'batch_names', cellstr(num2str([1:nBatches]'))});      % batch names
             
% Visualize the raw data (use 2 rows of 3 plots, 
% i.e. 6 tags per plot)
plot(batchX)

% Fit a PCA model to the data
A = 2;
batch_PCA = lvm({'X', batchX}, A);

% Let's see the score plots and Hotelling's T2 first
plot(batch_PCA)


% ---------
% DIAGNOSIS: there's a problem with batch 34 and 37.  
% ---------

% Batch 37: showed up in t_1
% --------
% Let's take a look when: right from the start of the batch
plot(batch_PCA, 'scores', {'batch', 37})

contrib_37 = contrib(batch_PCA, 37);
plot(contrib_37, 'scores', 1)  % which tags, and what 
% time periods, are related to problem?
% Plot the raw data to verify contributions
plot(batchX, 'highlight', 2, 3, 37)

% Does it also show up in SPEs (No)
plot(batch_PCA, 'spe', {'batch', 37})


% Batch 34: showed up in t_2
% --------
plot(batch_PCA, 'scores', {'batch', 34})

contrib_34 = contrib(batch_PCA, 34);
% which tags are related to batch 34, and at what times?
plot(contrib_34, 'scores', 2)  
% Verify in the raw data
plot(batchX, 'highlight', 2, 3, 34)   

% Does it also show up in SPEs?
% Yes: at time t=105, around when problem occurred
plot(batch_PCA, 'spe', {'batch', 34})

% What are the SPE contributions?
plot(contrib_34, 'spe')

% Other plots
plot(batch_PCA, 'summary') % default
plot(batch_PCA, 'obs')
plot(batchX, 'onebatch', 34)

% -------------------
% EXCLUDE and REBUILD
% -------------------

% Exclude the identified bad batches and make them 
% testing data
[batchXgood, test_X] = batchX.exclude(1, [34, 37]);


% Monitoring: code not available yet
% ----------
%batch_PCA = batch_PCA.monitoring_limits();
%test_data = batch_PCA.apply({'X', test_X}); 
%plot(test_data, 'obs')
%plot(test_data, 'obs', 'monitor')


% -------------------
% PLS MODELLING
% -------------------

% Let's do a PCA on the Y-space of the batch quality 
pcaY = lvm({'X', Y_data}, 2);

% Score: 34, 37 (are expected), but also see 36.  
% 48 and 52 are on the edge
plot(pcaY, 'scores')
% Batch 37 slightly over the limit
plot(pcaY, 'spe')

% Build a batch PLS model on the 50 good batches
Y_data_good = Y_data;
Y_data_good([34, 37],:) = [];  % exclude the 2 bad batches
pls = lvm({'X', batchXgood, 'Y', Y_data_good});




