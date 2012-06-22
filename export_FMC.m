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

phase_id = [];
batch_id = [];
for n = 1:59
    for p = 1:325
        if p <= 175
            phase_id(end+1) = 1;
        elseif  p <= 250
            phase_id(end+1) = 2;
        elseif p <= 325
            phase_id(end+1) = 3;
        end
        batch_id(end+1) = FMC.batch_names(n);
    end
end
phase_id = phase_id(:);
batch_id = batch_id(:);
csvwrite('trajectories_X.csv', [batch_id, phase_id, FMC.X])


% Final quality attributes (CQAs)
% --------------------------------
% Add labels when creating the block
Y = block(FMC.Y, {'col_labels', FMC.Ynames}, {'row_labels', FMC.batch_names}); 
Y = Y.exclude(1, missing_chemistry);
%plot(Y, {'layout', [2, 4]})

% Batch MB PCA and PLS models
% ----------------------------
batch_mbpca = lvm({'Z-chemistry', Zchem, 'Z-timing', Zop, 'Trajectories', X }, 3);
batch_mbpca.export('FMC_PCA');

batch_mbpls = lvm({'Z-chemistry', Zchem, 'Z-timing', Zop, ...
                   'Trajectories', X, 'Y', Y}, 2);
batch_mbpls.export('FMC_PLS');
