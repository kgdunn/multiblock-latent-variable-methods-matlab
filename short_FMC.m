
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

A = 3;
MBPLS_model = lvm({'Z', Z, 'X', X, 'Y', Y},A);
plot(MBPLS_model)


MBPLS_model.export('FMC_model');

% Apply the model on the raw data again
MBPLS_model.apply({Z, X});


% MANUAL VERSION OF GETTING THE PREDICTIONS
% -----------------------------------------
% Using the model: use the first row in Z and X to predict Y
row = 10;
raw_data_Z = Z.data(row,:);
raw_data_X = X.data(row,:);
raw_data_Y = Y.data(row,:);  % <---- this is what we are trying to predict


% Assemble the raw data from various sources into blocks (use a cell array)
raw_data = cell(MBPLS_model.B, 1);  % <--- raw data from PLC/OPC/IP21/etc
data     = cell(MBPLS_model.B, 1);  % <--- preprocessed data
raw_data{1} = raw_data_Z;
raw_data{2} = raw_data_X;


% Start by preprocessing the data in each block
for b = 1:MBPLS_model.B    
    % Centering
    data{b} = raw_data{b} - MBPLS_model.PP{b}.mean_center;
    % Scaling
    data{b} = data{b} .* MBPLS_model.PP{b}.scaling;
    % Apply block scaling = sqrt(number of columns in that block)
    block_scaling_factor = sqrt(MBPLS_model.K(b));
    data{b} = data{b} ./ block_scaling_factor;    
end
    
% Now we repeat the following steps for each component (A)

% Accumulate the scores for each block (rows), and for all components (cols)
block_scores = zeros(MBPLS_model.B, MBPLS_model.A);
super_scores = zeros(1, MBPLS_model.A);

for a = 1:MBPLS_model.A
    
    % Calculate the scores for each block:
    for b = 1:MBPLS_model.B
        
        % Basic calculation = (x * p')/(p'*p), but we will always assume there
        % is missing data, in which case the calculation is a bit a different.
        % p = loadings for each block in this case.
        
        keep = not(isnan(data{b}));  % these are entries that are NOT missing
        block_weights_a = MBPLS_model.W{b}(:,a);
        num = sum(data{b}(keep)' .* block_weights_a(keep));
        den = sum(block_weights_a(keep) .*  block_weights_a(keep));
        block_scores(b, a) = num / den;
    end
    
    % Now calculate the scores for the super level (the overall scores):
    super_weights_a = MBPLS_model.super.W(:,a);
    super_scores(a) = block_scores(:,a)' * super_weights_a;
    
    % Use the super scores to deflate each block:
    for b = 1:MBPLS_model.B
        block_loadings_a = MBPLS_model.P{b}(:,a);
        deflate = super_scores(a) * block_loadings_a';  % Note the transpose
        data{b} = data{b} - deflate;     % deflate the preprocessed data
    end
end

% Calculate SPE and T2 for each block, and the superblock:
SPE_block = zeros(MBPLS_model.B, 1);
T2_block = zeros(MBPLS_model.B, 1);
SPE_super = 0.0;
for b = 1:MBPLS_model.B
    keep = not(isnan(data{b})); % these are entries that are NOT missing
    SPE_block(b) = sum(data{b}(keep) .* data{b}(keep));
    SPE_super = SPE_super + SPE_block(b);  % <-- super level SPE = sum of all block SPEs
    
    
    S_matrix_b = MBPLS_model.stats{b}.S;
    T2_block(b) = block_scores(b,:) * S_matrix_b * block_scores(b,:)';
end
S_matrix_super = MBPLS_model.super.S;
T2_super = super_scores * S_matrix_super * super_scores'; % <-- super level T2


% Finally, calculate the predicted Y values: y_pred = T * C'
superlevel_C = MBPLS_model.super.C;    % M x A matrix; M = number of Y-variables
y_pred = super_scores * superlevel_C'; % (1 x A)(A x M) = 1 x M predictions

% Unscale predictions:
y_pred = y_pred ./ MBPLS_model.YPP.scaling;
% Uncenter predictions:
y_pred = y_pred + MBPLS_model.YPP.mean_center;



% CHECKS: compare our calculated values above to the values from the model,
% to make sure we are correct: should be vectors of zeros
super_scores - MBPLS_model.super.T(row, :)
for b = 1:MBPLS_model.B
    block_scores(b,:) - MBPLS_model.T{b}(row,:)
end





