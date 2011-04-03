
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

show_plots = false;
special_plots = false;
fontsize =14;

FMC = load('datasets/FMC.mat');

% Initial conditions block: operations
% -------------------------
Zop = block(FMC.Zop);
Zop.add_labels(1, FMC.batch_names);
Zop.add_labels(2, FMC.Zop_names)   % you can always add labels later on 
%plot(Zop) 

% Initial conditions block: chemistry
% -------------------------
Zchem = block(FMC.Zchem);
Zchem.add_labels(1, FMC.batch_names);
Zchem.add_labels(2, FMC.Zchem_names);
missing_chemistry = [12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 34, 35, 53];
Zchem = Zchem.exclude(1, missing_chemistry);
%plot(Zchem)

% Batch data block (pre-aligned)
% ------------------------------
X = block(FMC.X, 'X: batch data',...                     % name of the block
                 {'batch_tag_names', FMC.Xnames}, ...    % trajectory names
                 {'batch_names', FMC.batch_names});      % batch names
X = X.exclude(1, missing_chemistry);
temp = X.exclude(2, [2, 5]);
%plot(X, {'layout', [2, 3]})

% Final quality attributes (FQAs)
% --------------------------------
% Add labels when creating the block
Y = block(FMC.Y, {'col_labels', FMC.Ynames}, {'row_labels', FMC.batch_names}); 
Y.name = 'My name';
%plot(Y, {'layout', [2, 6]})


% Let's start with a PCA on the Y-block, to understand the quality variables
% We will use 3 components
fqa_pca = lvm({'FQAs', Y}, 1);
%plot(fqa_pca)

% There seem to be 2 clusters in the FQA space.  Let's take a look at
% contributions between points 



% Understand the effect of chemistry on the Y's
Y_copy = Y.copy();
Y_copy.exclude(1, missing_chemistry);
pls_chemistry = lvm({'Z-chemistry', Zchem, 'Y', Y_copy}, 1);


% Create monitoring model
% bad_batch = [3, 5, 6, 7, 34:71];
% 
% Z.exclude(1, bad_batch);
% Z.exclude(2, [1:8]);
% Zcopy.exclude(2, [1:8]);
% Y.exclude(1, bad_batch);
% X.data(bad_batch, :) = [];
% X.batch_raw(bad_batch, :) = [];
% 
% A_mon = 3;
% mon = lvm({'Z', Z, 'X', X, 'y', Y}, A_mon);

% SPEs are in disagreement: too high
% Scores for existing batches are offset slightly, or a lot in some cases



%N = mon.N;
if special_plots
    % Score plot
    hF = figure('Color', [1, 1, 1]);
    hT = axes;
    plot(mon.super.T(:,1), mon.super.T(:,2),'.')
    hold on
    %S = std(mon.super.T);
    s_i = 1./sqrt(mon.super.S(1,1));
    s_j = 1./sqrt(mon.super.S(2,2));
    plot_ellipse(s_i, s_j, mon.super.lim.T2(A_mon))
    grid()
    xlabel('t_1', 'FontSize', fontsize, 'Fontweight', 'bold')
    ylabel('t_2', 'FontSize', fontsize, 'Fontweight', 'bold')
    
    % SPE plot
    hF = figure('Color', [1, 1, 1]);
    hS = axes;
    plot(mon.super.SPE(:,A_mon), '.-')
    hold on
    hS_lim = plot([0 N], [mon.super.lim.SPE(A_mon) mon.super.lim.SPE(A_mon)], 'r-', 'linewidth', 2)
    grid()
    xlabel('Batches', 'FontSize', fontsize, 'Fontweight', 'bold')
    ylabel('SPE', 'FontSize', fontsize, 'Fontweight', 'bold')
    
    
    % Where does batch 3 lie?
    which = 3;
    new = cell(1, 2);
    new{1} = Zcopy.data(which,:);
    new{2} = Xcopy.data(which,:);
    out = mon.apply(new);

    
    for n = 1:shape(Zcopy, 1)
        which = n;
        new = cell(1, 2);
        new{1} = Zcopy.data(which,:);
        new{2} = Xcopy.data(which,:);
        out = mon.apply(new);
        
        axes(hT)
        plot(hT, out.T_super(1), out.T_super(2), 'k.')
        text(hT, out.T_super(1)+0.1, out.T_super(2)+.1, num2str(n))
        
        plot(hS, n +N,  out.stats.SPE{A_mon}, 'k.')
        text(hS, n+N,   out.stats.SPE{A_mon}+0.1, num2str(n))
        set(hS_lim, 'XData', [0 N+n])
    end
end



