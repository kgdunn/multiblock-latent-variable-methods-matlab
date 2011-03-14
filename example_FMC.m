
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
Z.add_labels(2, FMC.Znames)
if show_plots
    plot(Z)
end

% Batch data block (pre-aligned)
tag_names = {'CTankLvl','DiffPres','DryPress','Power','Torque','Agitator', ...
             'J-Temp-SP','J-Temp','D-Temp-SP','D-Temp','ClockTime'};
         
X = block(FMC.batchSPCData, 'X block',...                       % name of the block
                            {'batch_tag_names', tag_names}, ... % tag names
                            {'batch_names', FMC.Xnames}); 
plot(X, {'layout', [2, 3]}, {'mark', 59})