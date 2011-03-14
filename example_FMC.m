
% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% FMC case study. 
% 
% References:
% 
% [1] http://dx.doi.org/10.1021/ie0300023, "Troubleshooting of an Industrial 
%                                  Batch Process Using Multivariate Methods"
% [2] http://dx.doi.org/10.1016/B978-044452701-1.00108-3, "Batch process 
%                                                         modeling and MSPC"

show_plots = false;

FMC = load('datasets/FMC.mat');

% Initial conditions block
Z = block(FMC.Z);
Z.add_labels(2, FMC.Znames)   % you can always add labels later on 
if show_plots
    plot(Z) %#ok<*UNRCH>
end

% Batch data block (pre-aligned)
tag_names = {'CTankLvl','DiffPres','DryPress','Power','Torque','Agitator', ...
             'J-Temp-SP','J-Temp','D-Temp-SP','D-Temp','ClockTime'};
         
X = block(FMC.batchSPCData, 'X block',...                       % name of the block
                            {'batch_tag_names', tag_names}, ... % tag names
                            {'batch_names', FMC.Xnames}); 

if show_plots
    plot(X, {'layout', [2, 3]}, {'mark', 59})
end

% Final quality attributes (FQAs)
Y = block(FMC.Y, {'col_labels', FMC.Ynames});   % Add labels when creating the block
if show_plots
    plot(Y)
end

% Let's start with a PCA on the Y-block, to understand the quality variables
% We will use 3 components
fqa_pca = lvm({'FQAs', Y}, 3);

plot(fqa_pca.T{1}(:,1), fqa_pca.T{1}(:,2),'.'), grid

% There seem to be 2 clusters in the FQA space.  Let's take a look at
% contributions between points 
