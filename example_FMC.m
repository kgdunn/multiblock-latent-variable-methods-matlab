
% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
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

clear all
close all
FMC = load('datasets/FMC.mat');

% Initial conditions block: chemistry
% -------------------------
Zchem = block(FMC.Zchem);
Zchem.add_labels(1, FMC.batch_names);
Zchem.add_labels(2, FMC.Zchem_names);
missing_chemistry = [12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 34, 35, 53];
Zchem = Zchem.exclude(1, missing_chemistry);
%plot(Zchem)

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
    cqa_pca = lvm({'CQAs', Y}, 2);
    plot(cqa_pca)

    % There seem to be 2 clusters in the CQA space.  Take a look at contributions.
    % To confirm contributions in the raw data:
    plot(Y, {'mark', '61'})
    plot(Y, {'mark', '14'})
end


% Understand the effect of chemistry on the Y's (PLS)
% -----------------------------------
if true
    pls_chemistry = lvm({'Z-chemistry', Zchem, 'Y', Y}, 2);
    plot(pls_chemistry)
end

% Understand the effect of operating conditions on the Y's  (PLS)
% -----------------------------------
if true
    pls_operating = lvm({'Z-timing', Zop, 'Y', Y}, 2);
    plot(pls_operating)
end


% Multiblock PLS model: effect of chemistry and operating conditions on the Y's
% --------------------
if true
    pls_mb = lvm({'Z-chemistry', Zchem, 'Z-timing', Zop, 'Y', Y}, 3);
    plot(pls_mb)
    plot(Zchem, {'mark', '20'});
end
if true    
    Zcombined = block([FMC.Zchem FMC.Zop]);
    Zcombined.add_labels(1, FMC.batch_names);
    Zcombined.add_labels(2, [FMC.Zchem_names; FMC.Zop_names]);
    missing_chemistry = [12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 34, 35, 53];
    Zcombined = Zcombined.exclude(1, missing_chemistry);
    %plot(Zcombined)

    pls_combinedZ = lvm({'Z-combined', Zcombined, 'Y', Y}, 2);
    plot(pls_combinedZ)
    plot(pls_combinedZ, {'mark', '20'});    
end


% Take a look only at the trajectories
% ------------------------------------
if true
    batchPCA = lvm({'Trajectories', X}, 2);
    plot(batchPCA)
    plot(X, {'mark', '20'});
end

% And the trajectories vs the Y
% ------------------------------------
if true
    batchPLS = lvm({'Trajectories', X, 'Y', Y}, 2);
    plot(batchPLS)
    plot(X, {'mark', '42'});
end



% Batch MB PLS model
% -------------------
if true
    batch_mbpls = lvm({'Z-chemistry', Zchem, 'Z-timing', Zop, ...
                       'Trajectories', X, 'Y', Y}, 2);
    plot(batch_mbpls)
end

