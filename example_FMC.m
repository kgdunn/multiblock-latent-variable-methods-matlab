% Copyright (c) 2010-2012 ConnectMV, Inc. All rights reserved.
% Licensed under the BSD license.
% -------------------------------------------------------------------------
%
% FMC case study. 
% 
% References:
% 
% [1] http://dx.doi.org/10.1021/ie0300023, 
%      "Troubleshooting of an Industrial Batch Process Using Multivariate Methods"
% [2] http://dx.doi.org/10.1016/B978-044452701-1.00108-3, 
%      "Batch process modeling and MSPC"
% [3] http://digitalcommons.mcmaster.ca/opendissertations/1596/
%      "Batch Process Improvement Using Latent Variable Methods"

close all
FMC = load(['datasets', filesep, 'FMC.mat']);

% Initial conditions block: chemistry
% -------------------------
Zchem = block(FMC.Zchem);
Zchem.add_labels(1, FMC.batch_names);
Zchem.add_labels(2, FMC.Zchem_names);
missing_chemistry = [];%12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 34, 35, 53];
Zchem = Zchem.exclude(1, missing_chemistry);

% Initial conditions block: operations
% -------------------------
Zop = block(FMC.Zop);
Zop.add_labels(1, FMC.batch_names);
Zop.add_labels(2, FMC.Zop_names)   % you can always add labels later on 
Zop = Zop.exclude(1, missing_chemistry);
%plot(Zop) 

% Batch data block (pre-aligned)
% ------------------------------
X = block(FMC.X, 'X: batch data',...                     % name of the block
                 {'batch_tag_names', FMC.Xnames}, ...    % trajectory names
                 {'batch_names', FMC.batch_names});      % batch names
X = X.exclude(1, missing_chemistry);
%plot(X, {'layout', [2, 3]}, {'mark', {'28', '43'}})
%plot(X, {'layout', [2, 3]})

% Final quality attributes (CQAs)
% --------------------------------
% Add labels when creating the block
Y = block(FMC.Y, {'col_labels', FMC.Ynames}, {'row_labels', FMC.batch_names}); 
Y = Y.exclude(1, missing_chemistry);
%plot(Y, {'layout', [2, 4]})


% Let's start with a PCA on the Y-block, to understand the quality variables
% We will use 2 components
% ----------------------------------
if false
    cqa_pca = lvm({'CQAs', Y}, 3);
    plot(cqa_pca)

    % There seem to be 2 clusters in the CQA space.  Take a look at contributions.
    % To confirm contributions in the raw data:
    plot(Y, {'mark', '61'})
    plot(Y, {'mark', '14'})
end


% Understand the effect of chemistry on the Y's (PLS)
% -----------------------------------
if false
    pls_chemistry = lvm({'Z-chemistry', Zchem, 'Y', Y}, 2);
    plot(pls_chemistry)
end

% Understand the effect of operating conditions on the Y's  (PLS)
% -----------------------------------
if false
    pls_operating = lvm({'Z-timing', Zop, 'Y', Y}, 2);
    plot(pls_operating)
    plot(Zop, {'mark', '20'})
end

% Multiblock PLS model: effect of chemistry and operating conditions on the Y's
% --------------------
if false
    pls_mb = lvm({'Z-chemistry', Zchem, 'Z-timing', Zop, 'Y', Y}, 2);
    plot(pls_mb)
    plot(Zchem, {'mark', '20'});
end
if false
    Zcombined = block([FMC.Zchem FMC.Zop]);
    Zcombined.add_labels(1, FMC.batch_names);
    Zcombined.add_labels(2, [FMC.Zchem_names; FMC.Zop_names]);
    missing_chemistry = [12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 34, 35, 53];
    Zcombined = Zcombined.exclude(1, missing_chemistry);
    %plot(Zcombined)

    pls_combinedZ = lvm({'Z-combined', Zcombined, 'Y', Y}, 2);
    plot(pls_combinedZ)
end

% Take a look only at the trajectories
% ------------------------------------
if false
    batchPCA = lvm({'Trajectories', X}, 2);
    plot(batchPCA)
    plot(X, {'mark', '20'});
end

% And the trajectories vs the Y
% ------------------------------------
if false
    batchPLS = lvm({'Trajectories', X, 'Y', Y}, 2);
    plot(batchPLS)
    plot(X, {'mark', '13'});
    plot(X, {'mark', '5'});
    plot(X, {'mark', '7'});
end


% Batch MB PLS model
% -------------------
if true
    batch_mbpls = lvm({'Z-chemistry', Zchem, 'Z-timing', Zop, ...
                       'Trajectories', X, 'Y', Y}, 2);
    plot(batch_mbpls)
end

% Export the model to use with SVM
% ---------------------------------
% Model's super scores are the SVM features
features = batch_mbpls.super.T;  
% Class designation: 1=On-specification Y's
%                    2=Off-specification Y's
%                    3=Good Y, but high residual solvent
class = [1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1 ...
         1     1     1     1     1     1     1     1     2     2     2     2     2     2     2     2     2     2     2     2 ...
         2     2     2     2     2     2     2     2     2     2     2     2     3     3     3     3     3     3     3]';
%xlswrite('FMC_SVM_features.xls', [class features])

% Build a SVM classifier for 1 vs 2 (on-spec vs off-spec)
labels_train = (class==1) + (class==2);
labels_train(class==1)= -1;
labels_train(class==2)= +1;
data_train = features(labels_train ~= 0, :);
labels_train(labels_train == 0) = [];

svm_model_2 = train_cv_svm(data_train, labels_train);

% Plot the mesh of "c" vs "g" for the RBF parameters
figure; mesh(svm_model_2.cv_results)


