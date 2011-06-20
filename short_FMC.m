
% Shortened FMC data set
% ----------------------
FMC = load('datasets/FMC.mat');

% Only use the first 10 rows in all blocks
use_rows = 1:10;

% Using only 3 tags in the Z-block
use_Z_vars = [1, 5, 7];

% Using only 5 tags in the batch block
use_batch_vars = [1,6, 8, 10, 11]; 

% Use a subset of the variables in Y
use_Y_vars = [1, 3, 5, 6];

Z = FMC.Zop(use_rows, use_Z_vars);
X = FMC.X(:, use_batch_vars);
Y = FMC.Y(use_rows, use_Y_vars);

% Now create block objects
Z = block(Z);
X = block(X, {'batch_tag_names', FMC.Xnames(use_batch_vars)}, ...
             {'nbatches', 59});
[X_throwaway, X] = X.exclude(1, use_rows);
Y_column_names = cellstr(FMC.Ynames);
Y_column_names = Y_column_names(use_Y_vars);
Y = block(Y, {'col_labels', Y_column_names});

A = 2;
MBPLS_model = lvm({'Z', Z, 'X', X, 'Y', Y},A);
plot(MBPLS_model)


MBPLS_model.export('FMC_model')
