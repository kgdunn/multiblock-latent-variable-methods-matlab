
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
special_plots = false;
fontsize =14;

FMC = load('datasets/FMC-full.mat');

% Initial conditions block
Z = block(FMC.Z);
Zcopy = copy(Z);
Z.add_labels(2, FMC.Znames)   % you can always add labels later on 
if show_plots
    plot(Z) %#ok<*UNRCH>
end

% Batch data block (pre-aligned)
tag_names = {'CTankLvl','DiffPres','DryPress','Power','Torque','Agitator', ...
             'J-Temp-SP','J-Temp','D-Temp-SP','D-Temp','ClockTime'};
         
X = block(FMC.X, 'X block',...                       % name of the block
                            {'batch_tag_names', tag_names}, ... % tag names
                            {'batch_names', 1:71}); 
Xcopy = copy(X);
if show_plots
    plot(X, {'layout', [2, 3]})
end

% Final quality attributes (FQAs)
Y = block(FMC.Y, {'col_labels', FMC.Ynames});   % Add labels when creating the block
Ycopy = copy(Y);
if show_plots
    plot(Y, {'layout', [2, 6]})
end

if special_plots
    % Let's start with a PCA on the Y-block, to understand the quality variables
    % We will use 3 components
    fqa_pca = lvm({'FQAs', Y}, 2);
    hF = figure('Color', [1, 1, 1]);
    plot(fqa_pca.stats{1}.SPE(:,2))
    hold on
    lim = fqa_pca.lim{1}.SPE(:,2);
    plot([0 71], [lim, lim], 'r')
    axis tight
    grid

    plot(fqa_pca.T{1}(:,1), fqa_pca.T{1}(:,2),'.'), grid
end
% There seem to be 2 clusters in the FQA space.  Let's take a look at
% contributions between points 

% Create monitoring model
bad_batch = [3, 5, 6, 7, 34:71];

Z.exclude(1, bad_batch)
Z.exclude(2, [1:8])
Y.exclude(1, bad_batch)
X.data(bad_batch, :) = [];
X.batch_raw(bad_batch, :) = [];

A_mon = 2;
mon = lvm({'Z', Z, 'X', X, 'y', Y}, A_mon);

if special_plots
    % Score plot
    hF = figure('Color', [1, 1, 1]);
    plot(mon.super.T(:,1), mon.super.T(:,2),'.')
    hold on
    %S = std(mon.super.T);
    s_i = 1./sqrt(mon.super.S(1,1));
    s_j = 1./sqrt(mon.super.S(2,2));
    plot_ellipse(s_i, s_j, mon.super.lim.T2(A_mon))
    grid()
    xlabel('t_1', 'FontSize', fontsize, 'Fontweight', 'bold')
    ylabel('t_2', 'FontSize', fontsize, 'Fontweight', 'bold')
end

% Batch 3 lie?
which = 3;
new = cell(1, 2);
new{1} = Z.data(3,:);
new{2} = X.data(3,:);

out = mon.apply(new);

