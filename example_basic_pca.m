% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------

% Load the data
% -------------
quality = load('datasets/LDPE-quality.mat');
tag_names = {'Conversion', 'Mn', 'Mw', 'LCB', 'SCB'};

% Create the "X" block: 
% -------------------------
quality = block(quality.X); 
quality.N                          % Answer is 54: indicates 54 rows in the dataset
quality.add_labels(1, 1:54);       % Just add numbers for the rows (1)
quality.add_labels(2, tag_names);  % Add labels for the columns (2)


% Build a PCA on the Y-block, to understand the critical to quality attributes
% ----------------------------------
cqa_pca = lvm({'CQAs', quality}, 2);
plot(cqa_pca)

% Observations 16, 17 and 44 are outliers in SPE: exclude them (rows --> 1)
[quality, test_data] = quality.exclude(1, [16, 17, 44]);
quality.N  % answer is 51

cqa_pca_updated = lvm({'CQAs', quality}, 2);
plot(cqa_pca_updated)


% Use the testing data
test_data.N  % answer is 3, because we excluded 3 rows

% The testing data must be provided in the same way as the training data
output = cqa_pca_updated.apply({test_data});

% Verify their SPE values are too high:
output.stats.SPE{1}

