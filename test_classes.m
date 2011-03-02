X_raw = [3, 4, 2, 2; 4, 3, 4, 3; 5.0, 5, 6, 4];
PCA_model_1 = lvm({'X', X_raw}, 1);

% Load the raw data
dupont = load('../code/datasets/DuPont.mat');
data = struct;
data.X = dupont.tech;

A = 3;
batch_PCA = lvm({'X', data.X}, A);

% Let's see the score plots and Hotelling's T2 first
plot(batch_PCA, 'scores')

% Understand where and when the variance is explained
% Pretty much all variables, for the entire batch show 
%in LV1 (lowest is tag 6)
plot(batch_PCA, 'R2')


% ---------
% DIAGNOSIS
% ---------

% Batch 54: high t_1 the entire batch
% --------
plot(batch_PCA, 'scores', {'batch', 54})

% Let's first understand the loadings: p_1
plot(batch_PCA.X, 'loadings', 1)

% We should pick it up on a contribution plot as well
contrib_54 = contrib(batch_PCA, 54);
plot(contrib_54, 'scores', 1)

% Plot the raw data to see problem
plot(batch_X, 'highlight', 2, 5, 54)


% Batch 53 (and 55 also in t2)
plot(batch_PCA.X, 'loadings', 2)   % loadings for p_2
contrib_53 = contrib(batch_PCA, 53);
plot(contrib_53, 'scores', 2)
plot(batch_X, 'highlight', 2, 5, 53)

% batch 49: shows on SPE 
% --------

% Overall SPE
plot(batch_PCA, 'spe')

% Instantaneous SPE
plot(batch_PCA, 'spe', {'batch', 49})

% See it in a contribution plot also
contrib_49 = contrib(batch_PCA, 49);
plot(contrib_49, 'spe')

% Finally, verify the problem exists in the raw data 
% for batch 49
plot(batch_X, 'highlight', 2, 5, 49)


% -------------------
% EXCLUDE and REBUILD
% -------------------

% Exclude the identified bad batches and make them 
% testing data
[batch_X, test_X] = batch_X.exclude(1, [49, 50:55]);

% See now how more consistent the trajectories are
plot(batch_X, 'raw', 2, 5) 

% All the unusual trajectories are in the testing data
plot(test_X, 'raw', 2, 5)

% Rebuild the model on the cleaned data
batch_PCA_update = lvm({'X', batch_X}, A);

% Check the score plot
plot(batch_PCA_update, 'scores')
plot(batch_PCA_update, 'spe')

% And also see where the variability lies: in which 
% variables and at what times
plot(batch_PCA_update, 'R2')

% Understand the p_3 direction that shows a group of 
% outliers
plot(batch_PCA_update.X, 'loadings', 3)

% See batch 39 in the scores trajectories and contributions
plot(batch_PCA_update, 'scores', {'batch', 39})
contrib_39 = contrib(batch_PCA_update, 39);
plot(contrib_39, 'scores', 3)

% Plot the raw data to see problem
plot(batch_X, 'highlight', 2, 5, 39)

% Batch 45 was known to give bad product (info from DuPont)
% This analysis confirms it.
plot(batch_X, 'highlight', 2, 5, 45)
contrib_45 = contrib(batch_PCA_update, 45);
plot(contrib_45, 'scores', 3)

% Look at the t_2 direction, batch 37, 48, 44
% So understand what p_2 is modelling
plot(batch_PCA_update.X, 'loadings', 2)


% Check the contribution plot for batch 37 in the 
% t2 direction
contrib_37 = contrib(batch_PCA_update, 37);
plot(contrib_37, 'scores', 2)

% Verify it in the raw data
plot(batch_X, 'highlight', 2, 5, 37)


% Apply the model to the testing data
applied = batch_PCA_update.apply({'X', test_X});
%plot(applied.blocks{1}.T(:,1), T(:,2))


% ---------------------------
% EXCLUDE and REBUILD AGAIN
% ---------------------------
% Exclude the identified bad batches and make 
% them testing data
[batch_X, test_X] = batch_X.exclude(1, [37, 39, 43:48]);
batch_PCA3 = lvm({'X', batch_X}, A);
plot(batch_PCA3, 'scores')
plot(batch_PCA3, 'spe')

plot(batch_PCA3.X, 'loadings', 1)
plot(batch_PCA3, 'R2')

plot(batch_X, 'highlight', 2, 5, 8)
