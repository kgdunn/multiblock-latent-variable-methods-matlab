function test_plots(varargin)
    test_basic_plots()
end


function test_basic_plots()
    % plot(model)                        % summary plot of T2, SPE, R2X per PC
    % > plot(model, 'scores')              % This line and the next do the same
    % > plot(model, 'scores', 'overall')   % All scores for overall block
    % > plot(model, 'scores', {'block', 1})% All scores for block 1
    % > plot(model, {'scores', 2}, {'block', 1})  % t_2 scores for block 1
    %
    % Scores for block 1 (a batch block), only showing batch 38
    % > plot(model, 'scores', {'block', 1}, {'batch', 38)  


    FMC = load('datasets/FMC.mat');
    
    % Initial conditions
    Z = block(FMC.Z);
    Z.add_labels(2, FMC.Znames)   % you can always add labels later on 


    % Final quality attributes (FQAs)
    Y = block(FMC.Y, {'col_labels', FMC.Ynames});   % Add labels when creating the block
    
    % Let's start with a PCA on the Y-block, to understand the quality variables
    % We will use 3 components
    fqa_pca_Y = lvm({'FQAs', Y}, 2);

    % We should see 2 clusters in the scores
    %plot(fqa_pca.T{1}(:,1), fqa_pca.T{1}(:,2),'.'), grid    
    plot(fqa_pca_Y, 'scores')               % All scores for overall block
    plot(fqa_pca_Y, 'scores', {'block', 1}) % All scores for block 1
    plot(fqa_pca_Y, 'scores', {'block', 'fqas'}) % All scores for block 1
    plot(fqa_pca_Y, {'scores', 2}, {'block', 1})  % t_2 scores for block 1
    plot(fqa_pca_Y, 'loadings')
    plot(fqa_pca_Y, {'loadings', 1})
    plot(fqa_pca_Y, {'SPE'})
    plot(fqa_pca_Y, 'T2')
    plot(fqa_pca_Y, 'R2')
    
    
    % Multiblock model with 2 blocks
    fqa_pca_ZY = lvm({'Initial', Z, 'FQA', Y}, 2);
    
    plot(fqa_pca_ZY, 'scores')
    plot(fqa_pca_ZY, 'scores')   % All scores for overall block
    plot(fqa_pca_ZY, 'scores', {'block', 1})% All scores for block 1
    plot(fqa_pca_ZY, {'scores', [2,3]}, {'block', 1})  % t_2 scores for block 1
    plot(fqa_pca_ZY, {'scores'}, {'block', 1})  % t_2 scores for block 1
    plot(fqa_pca_ZY, {'scores'}, {'block', 2})  % t_2 scores for block 1
    
    
    
    
    % Batch model plots
    tag_names = {'CTankLvl','DiffPres','DryPress','Power','Torque','Agitator', ...
                 'J-Temp-SP','J-Temp','D-Temp-SP','D-Temp','ClockTime'};

    X = block(FMC.batchSPCData, 'X block',...                       % name of the block
                                {'batch_tag_names', tag_names}, ... % tag names
                                {'batch_names', FMC.Xnames}); 
    
    % Batch PCA model
    fqa_pca_X = lvm({'Batch', X}, 3);
    
    % Scores for block 1 (a batch block), only showing batch 38    
    plot(fqa_pca_X, 'scores')
    plot(fqa_pca_X, 'scores')   % All scores for overall block
    plot(fqa_pca_X, 'scores', {'block', 1})% All scores for block 1
    plot(fqa_pca_X, {'scores', 2}, {'block', 1})  % t_2 scores for block 1
    plot(fqa_pca_X, {'scores'}, {'block', 1})  % t_2 scores for block 1
    plot(fqa_pca_X, {'scores'}, {'block', 2})  % t_2 scores for block 1
    plot(fqa_pca_X, 'scores', {'block', 1}, {'batch', 38})
    plot(fqa_pca_X, 'scores', {'block', 1}, {'batch', 38}, {'abc', 123})
    plot(fqa_pca_X, 'loadings')
    plot(fqa_pca_X, {'loadings', 1})
    plot(fqa_pca_X, {'SPE'})
    plot(fqa_pca_X, 'T2')
    plot(fqa_pca_X, 'R2')
    
end