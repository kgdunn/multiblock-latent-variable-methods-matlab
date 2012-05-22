% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% Licensed under the BSD license.
% -------------------------------------------------------------------------

% Load the raw data
data = load(['datasets', filesep, 'SBR.mat']);
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

% Let's take a look at the model first: scores and SPE
%plot(batch_PCA)

% Highlight batch 34 and 37 (scores) and batch 8 (SPE)
%plot(batchX, {'layout', [2, 4]}, {'mark', {'34'}})
%plot(batchX, {'layout', [2, 4]}, {'mark', {'37'}})
%plot(batchX, {'layout', [2, 4]}, {'mark', {'8'}})



% Exclude the identified bad batches and make them 
% testing data
[X_good, test_X] = batchX.exclude(1, [34, 37]);


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
Y = block(Y_data);
Y.add_labels(1, cellstr(num2str([1:nBatches]')));
Y.add_labels(2, cellstr(data.Ynames));
plot(Y)
pcaY = lvm({'Quality', Y}, 2);

% Score: 34, 37 (are expected), but also see 36.  
% 48 and 52 are on the edge

% Build a batch PLS model on the 50 good batches
[Y_good, Y_test] = Y.exclude(1, [34, 37]);
pls = lvm({'X', X_good, 'Y', Y_good}, 2);
plot(pls)


