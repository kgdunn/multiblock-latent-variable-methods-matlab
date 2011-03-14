function unit_tests(varargin)
    close all;
    
    test_significant_figures()
    
    
    
    PLS_batch_data();
    
    
    % PCA tests
    Wold_article_PCA_test()
    PCA_no_missing_data()
    PCA_with_missing_data()
    PCA_batch_data()
    
    PCA_cross_validation_no_missing()
    MBPCA_tests()
    
    % PLS tests
    basic_PLS_test()
    PLS_no_missing_data()
    PLS_with_missing_data() 
    
    
    MBPLS_tests();
    %PLS_randomization_tests()
    
    % Basic plots
    test_plots()
    
    % External testing file
    test_blocks()
        
    
    % TODO(KGD):
    % -----------
    % - SPE limits for a large data set: do 95% of data fall below limit?  Use qq-plot to verify

    % Implement this test: http://fseoane.net/blog/2011/computing-the-vector-norm/
    
    % TODO(KGD):
    % - Syntax tests
    % 1a) PCA = lvm({'X', X}, A);  quick build with A components
    % 1b) Same, except X is a block, not a matrix
    % 2)  options = lvm_opt(); 
    %     PCA_model = lvm({'X', X}, options);
    %   a)options.build_now = false; <-- used for cross-validation internally
    %   b)options.build_now = true;
    
    % Add test on this PLS model:
    % since it has 4 X-space variables, but there is no variance left in the
    % 4th PC to support another component (DOE on recipe data)
    %     lowarp = xlsread('tests/lowarp-data.xls');
    %     X = lowarp(:,1:4);
    %     Y = lowarp(:,5:end);
    %     options = lvm_opt(); 
    %     options.randomize_test.use = true;
    %     PLS_model = lvm({'X', X, 'Y', Y}, options);
    
    % Tests based on data in http://dx.doi.org/10.1002/cem.1248
    
    % TODO: tests based on values in http://dx.doi.org/10.1002/aic.690400509
    
    % TODO: tests based on Westerhuis MB PLS paper on the LDPE data set (p 313)
    
    % TODO: PLS tests and MBPLS tests from MacGregor and Jaeckle LDPE data set
    %       has the R2 values for PLS and MBPLS on p 829.
    
    % TODO: test for excluding rows and columns from ordinary and batch data
    %       sets.

    disp('ALL TESTS PASSED - if no errors were displayed.')
return

function X = create_orthogonal_matrix(rows, cols)
    % Creates an orthogonal matrix ``X`` of shape ``rows`` and ``cols`` so
    % that X' * X gives back a cols by cols identity matrix.  Note that
    % ``cols`` must be smaller or equal to ``rows``.
    %
    % Reference: http://dx.doi.org/10.1137/0908055 (also a technical report
    % from Stanford University, Technical Report No 6, August 1985:
    % http://statistics.stanford.edu/~ckirby/techreports/NSF/AND%20NSF%2006.pdf
    % Also see: www.jstor.org/stable/2346957
    
    assertTrue(cols <= rows);
    Y = randn(rows, rows);
    [X,r] = qr(Y);
    X = X(:,1:cols);    
    
return

function X = create_centered_decreasing_spread(rows, cols, spread, tol)
    % Creates a matrix of shape ``rows`` by ``cols`` where the columns are 
    % in decreasing variance: i.e. var(X(:,1)) > var(X(:,2)) etc.  The 
    % columns are mean centered. 
    % 
    % The ``spread`` vector must contain the desired standard deviation for
    % each column; there must be ``cols`` entries in the ``spread`` input.
    %
    % The ``tol`` input indicates how close the desired spread should be to
    % the actual spread.  Usually set this to 0.1.
    % 
    % The output matrix has *approximately* the desired spread
    assertTrue(cols == numel(spread));
    X = zeros(rows, cols);
    for k = 1:cols
        temp = randn(rows, 1) * spread(k);
        while abs(std(temp) - spread(k)) > tol
            temp = randn(rows, 1) * spread(k);
        end        
        X(:,k) = temp - mean(temp);
    end
return

function test_significant_figures()
% Ensures that the tests for significant figures are correct.
% TODO(KGD): come back to this test still.
    assertEAE(0.5412, 0.5414, 3)
    assertExceptionThrown('assertEAE:tolExceeded', ...
                            @assertEAE, 0.5412, 0.5414, 4)
    assertEAE(1.5412E-5, 1.5414E-5, 4)
    assertExceptionThrown('assertEAE:tolExceeded', ...
                @assertEAE, 1.5412E-5, 1.5414E-5, 5)
    %1.5412 == 1.5414       is True if sig_figs = 4, but False if sig_figs = 5
    %1.5412E-5 == 1.5414E-5 is True if sig_figs = 4, but False if sig_figs = 5
    %1.5412E+5 == 1.5414E+5 is True if sig_figs = 4, but False if sig_figs = 5
return

function Wold_article_PCA_test()
%   Tests from the PCA paper by Wold, Esbensen and Geladi, 1987
%   Principal Component Analysis, Chemometrics and Intelligent Laboratory
%   Systems, v 2, p37-52; http://dx.doi.org/10.1016/0169-7439(87)80084-9
    fprintf('PCA tests from literature: ');
    X_raw = [3, 4, 2, 2; 4, 3, 4, 3; 5.0, 5, 6, 4];
    X = block(X_raw);
    options = lvm_opt();     
    options.show_progress = false; 
    options.min_lv = 1;    
    PCA_model_1 = lvm({'X', X}, options);
    assertTrue(strcmp(PCA_model_1.blocks{1}.name, 'X'))

    X = block(X_raw);
    options = lvm_opt();     
    options.show_progress = false; 
    options.min_lv = 2;
    PCA_model_2 = lvm({'X', X}, options);

    % The mean centering vector should be [4, 4, 4, 3], page 40
    assertTrue(all(PCA_model_2.PP{1}.mean_center == [4, 4, 4, 3]));

    % The (inverted) scaling vector [1, 1, 0.5, 1], page 40
    assertTrue(all(PCA_model_2.PP{1}.scaling == [1, 1, 0.5, 1]));
    
    % With 2 components, the loadings are, page 40
    %  P.T = [ 0.5410, 0.3493,  0.5410,  0.5410],
    %        [-0.2017, 0.9370, -0.2017, -0.2017]])
    P = PCA_model_2.P{1};
    assertEAE(P(:,1), [0.5410, 0.3493, 0.5410, 0.5410]', 2);
    assertEAE(P(:,2), [-0.2017, 0.9370, -0.2017, -0.2017]', 2);

    T = PCA_model_2.T{1};
    assertEAE(T(:,1), [-1.6229, -0.3493, 1.9723]', 2)
    assertEAE(T(:,2), [0.6051, -0.9370, 0.3319]', 2)
    
    
    % R2 values, given on page 43
    R2b_a = PCA_model_2.stats{1}.R2b_a;
    assertEAE(R2b_a, [0.831, 0.169], 2)

    % SS values, on page 43
    SS_X = ssq(PCA_model_2.data, 1);
    assertEAE(SS_X, [0.0, 0.0, 0.0, 0.0], 3)

    % The remaining sum of squares, on page 43
    SS_X = ssq(PCA_model_1.data, 1);
    assertEAE(SS_X, [0.0551, 1.189, 0.0551, 0.0551], 3)
    
    
    % Superblock VIP's for single-block models = 1.0
    assertEAE(PCA_model_1.super.stats.VIP(1), 1.0, 8);
    assertEAE(PCA_model_2.super.stats.VIP(1), 1.0, 8);
    assertEAE(PCA_model_2.super.stats.VIP(2), 1.0, 8);
    
    % These are relative to ProSensus Multivariate Trial version, 2010, Revision 302
    assertEAE(PCA_model_1.stats{1}.VIP_a(:,1)', [1.082, 0.6987, 1.082, 1.082], 4)
    assertEAE(PCA_model_2.stats{1}.VIP_a(:,1)', [1.082, 0.6987, 1.082, 1.082], 4)
    assertEAE(PCA_model_2.stats{1}.VIP_a(:,2)', [1.0, 1.0, 1.0, 1.0], 4)

    % Cannot test against ProSensus T2: they divide by n-1 to calculate the
    % T-scores variance-covariance matrix. We divide by n.
    %
    %assertEAE(PCA_model_1.super.T2(:,1), [0.792655, 0.036726, 1.1706]', 4)
    %assertEAE(PCA_model_2.super.T2(:,1), [0.792655, 0.036726, 1.1706]', 4)
    %assertEAE(PCA_model_2.super.T2(:,2), [1.33333, 1.33333, 1.33333]', 4)
    
    % ProSensus Multivariate defines SPE = e'*e, where as we define it as 
    % sqrt(e'*e / K).  The values here have been scaled to undo this effect.
    ProMV_values = [0.366107, 0.877964, 0.110178];
    ProMV_values = sqrt(ProMV_values ./ 4);
    assertEAE(PCA_model_1.super.SPE(:,1), ProMV_values', 4)
    assertEAE(PCA_model_2.super.SPE(:,1), ProMV_values', 4)    
    assertEAE(PCA_model_2.super.SPE(:,2), [0, 0, 0]', 4)
    
    % Statistical limits
    assertEAE(PCA_model_1.super.lim.T2, 24.684, 3)
    assertEAE(PCA_model_1.super.lim.SPE, sqrt(1.2236/4), 4)
    assertEAE(PCA_model_1.super.lim.t, 7.8432, 4)
    
    
    % Testing data.  2 rows of new observations.
    X_test_raw = [3, 4, 3, 4; 1, 2, 3, 4.0];
    X_test = block(X_test_raw);
    assertEAE(X_test.data, [3, 4, 3, 4; 1, 2, 3, 4.0],5);
    
    % Apply the new data to an existing model
    testing_type_A = PCA_model_1.apply({X_test});      % send in a block variable
    testing_type_B = PCA_model_1.apply({X_test_raw});  % send in a raw array 
    assertEAE(testing_type_A.T{1}, [-0.2705, -2.0511]', 4)
    assertEAE(testing_type_B.T{1}, [-0.2705, -2.0511]', 4)
    
    % Extract a second component
    X_test_raw = [3, 4, 3, 4; 1, 2, 3, 4.0];
    X_test = block(X_test_raw);
    assertEAE(X_test.data, [3, 4, 3, 4; 1, 2, 3, 4.0],5);
    
    testing_type_C = PCA_model_2.apply({X_test});      % send in a block variable
    testing_type_D = PCA_model_2.apply({X_test_raw});  % send in a raw array 
    assertEAE(testing_type_C.T{1}, [-0.2705, -2.0511; 0.1009, -1.3698]', 3)
    assertEAE(testing_type_D.T{1}, [-0.2705, -2.0511; 0.1009, -1.3698]', 3)
    
        
    % Applying the model to the training data should give identical results
    X_new = [3, 4, 2, 2; 4, 3, 4, 3; 5.0, 5, 6, 4];
    X_new_1 = PCA_model_1.apply({X_new});
    X_new_2 = PCA_model_2.apply({X_new});
    
    assertEAE(X_new_1.T{1}(:,1), [-1.6229, -0.3493, 1.9723]', 2)
    assertEAE(X_new_1.T_super(:,1), [-1.6229, -0.3493, 1.9723]', 2)
    assertEAE(X_new_2.T{1}(:,1), [-1.6229, -0.3493, 1.9723]', 2)
    assertEAE(X_new_2.T_super(:,1), [-1.6229, -0.3493, 1.9723]', 2)
    assertEAE(X_new_2.T{1}(:,2), [0.6051, -0.9370, 0.3319]', 2)
    assertEAE(X_new_2.T_super(:,2), [0.6051, -0.9370, 0.3319]', 2)
    
    % Cannot test against ProSensus T2: they divide by n-1 to calculate the
    % T-scores variance-covariance matrix. We divide by n.
    %
    %assertEAE(X_new_1.stats.super.T2(:,1), [0.792655, 0.036726, 1.1706]', 4)
    %assertEAE(X_new_1.stats.super.T2(:,1), [0.792655, 0.036726, 1.1706]', 4)
    %assertEAE(X_new_2.stats.super.T2(:,1), [1.33333, 1.33333, 1.33333]', 4)
    
    % ProSensus Multivariate defines SPE = e'*e, where as we define it as 
    % sqrt(e'*e / K).  The values here have been scaled to undo this effect.
    ProMV_values = [0.366107, 0.877964, 0.110178];
    ProMV_values = sqrt(ProMV_values ./ 4);
    
    assertEAE(X_new_1.stats.SPE{1}, ProMV_values', 4)
    assertEAE(X_new_1.stats.super.SPE(:,1), ProMV_values', 4)
    assertEAE(X_new_2.stats.SPE{1}, [0, 0, 0]', 4)
    fprintf('OK\n');
return

function basic_PLS_test()
    % Tests based on http://dx.doi.org/10.1002/0470845015.cpa012
    
    % Also code up the example in Geladi and Kowalski, Analytica Chimica Acta, 185, p 19-32, 1986
    % http://dx.doi.org/10.1016/0003-2670(86)80028-9
    % http://dx.doi.org/10.1016/0003-2670(86)80029-0
return

function PCA_no_missing_data()
% Tests a normal PCA model, no missing data.
% Tests it against a model build in Simca-P, version 11.5.0.0 (2006).
    LDPE = load('tests/LDPE-PCA.mat');
    raw_data = LDPE.data;  % Raw data
    exp_m = LDPE.model;    % Expected model
    
    % Test that mean centering and scaling are correct
    X = block(raw_data.blocks{1});
    X = X.preprocess();
    assertEAE(X.data, raw_data.scaled_blocks{1}, 4)
    
    % Build the PCA model with LVM.m
    % ------------------------------
    A = exp_m.A;
    options = lvm_opt();     
    options.show_progress = false; 
    options.min_lv = A;
    PCA = lvm({'X', X}, options);
    
    
    % Now test the PCA model
    % ----------------------
    % T-scores
    scores_col = 8:9;
    T = exp_m.observations{1}.data(:, scores_col);
    assertEAE(PCA.T{1}, T, 2);
    
    % T^2
    T2_col = 2;
    T2 = exp_m.observations{1}.data(:, T2_col);
    assertEAE(PCA.stats{1}.T2(:,PCA.A), T2, 4);
    
    % X-hat
    X_hat_col = 3:7;
    X_hat = exp_m.observations{1}.data(:, X_hat_col);
    X_hat_calc = PCA.T{1} * PCA.P{1}';
    assertEAE(X_hat_calc, X_hat, 2);
    
      % Loadings
    loadings_col = 1:2;
    P = exp_m.variables{1}.data(:, loadings_col);
    assertEAE(PCA.P{1}, P, 4);
    
    % R2-per variable(k)-per component(a)
    R2_col = 4:5;
    R2k_a = exp_m.variables{1}.data(:, R2_col);
    assertEAE(PCA.stats{1}.R2k_a, R2k_a, 4);  
    
return

function PLS_no_missing_data()
% Tests a normal PLS model, no missing data.
% Tests it against a model build in Simca-P, version 11.5.0.0 (2006).
    LDPE = load('tests/LDPE-PLS.mat');
    raw_data = LDPE.data;  % Raw data
    exp_m = LDPE.model;    % Expected model
    
    % Test that mean centering and scaling are correct
    X = block(raw_data.blocks{1});
    X = X.preprocess();
    assertEAE(X.data, raw_data.scaled_blocks{1}, 2)
    
    Y = block(raw_data.blocks{2});
    Y = Y.preprocess();
    assertEAE(Y.data, raw_data.scaled_blocks{2}, 4)
    assertEAE(shape(Y), [54, 5], 4)
    assertTrue(shape(Y, 1) == 54)
    assertTrue(shape(Y, 2) == 5)
    
    % Build the PCA model with LVM.m
    % ------------------------------
    A = exp_m.A;
    options = lvm_opt();     
    options.show_progress = false; 
    options.min_lv = A;
    PLS = lvm({'X', X, 'Y', Y}, options);


    % Now test the PLS model
    % ----------------------
    % T-scores
    scores_col = 3:8;
    T = exp_m.observations{1}.data(:, scores_col);
    assertEAE(PLS.T{1}, T, 2, true);
    
    % T^2
    T2_col = 2;
    T2 = exp_m.observations{1}.data(:, T2_col);
    assertEAE(PLS.stats{1}.T2(:,PLS.A), T2, 4);
    
    % Y-hat
    Y_hat_col = 7:11;
    Y_hat = exp_m.observations{2}.data(:, Y_hat_col);
    %_hat_calc = PLS..data_pred;
    %Y_hat_PP = PLS.blocks{2}.preprocess(block(Y_hat_calc));
    assertEAE(PLS.Yhat.data, Y_hat, 2);
    
    % W-Loadings
    loadings_col = 1:6;
    W = exp_m.variables{1}.data(:, loadings_col);
    assertEAE(PLS.W{1}, W, 3, true);
    
    % R2-per variable(k)-cumulative: X-space
    R2X_col = 24;
    R2kX_cum = exp_m.variables{1}.data(:, R2X_col);
    assertEAE(sum(PLS.stats{1}.R2Xk_a(:,1:end),2), R2kX_cum, 4); 
    
    % R2-per variable(k)-per component(a): Y-space
    R2Y_col = 8;
    R2kY_cum = exp_m.variables{2}.data(:, R2Y_col);
    assertEAE(sum(PLS.super.stats.R2Yk_a, 2), R2kY_cum, 4); 
    
    % TODO(KGD): Coefficients

return

function PCA_with_missing_data()
    warning('off', 'MATLAB:xlsread:Mode')
    warning('off', 'MATLAB:xlsread:ActiveX')
    kamyr = xlsread('tests/kamyr-digester-subset.xls');
    warning('on', 'MATLAB:xlsread:Mode')
    warning('on', 'MATLAB:xlsread:ActiveX')
    kamyr = kamyr(:,2:end);
    
    % Test that mean centering and scaling are correct
    X = block(kamyr);
    X = X.preprocess();
    
    % Build the model
    A = 2;
    options = lvm_opt();     
    options.show_progress = false; 
    options.min_lv = A;
    PCA = lvm({'Column', X}, options);
    
    assertEAE(PCA.P{1}' * PCA.P{1} - eye(A), zeros(A), 1);
    
    % TODO(KGD): complete tests: check limits also.    
return

function PLS_with_missing_data()
    warning('off', 'MATLAB:xlsread:Mode')
    warning('off', 'MATLAB:xlsread:ActiveX')
    kamyr = xlsread('tests/kamyr-digester-subset.xls');
    warning('on', 'MATLAB:xlsread:Mode')
    warning('on', 'MATLAB:xlsread:ActiveX')
    kamyr = kamyr(:,2:end);
    
    % Test that mean centering and scaling are correct
    X = block(kamyr(:, 2:end));
    X = X.preprocess();
    Y = block(kamyr(:, 1));
    Y = Y.preprocess();
    
    % Build the model
    A = 2;
    PLS = lvm({'X', X, 'Y', Y}, A);
    
    % TODO(KGD): complete tests: check limits also.    
return

function PCA_batch_data()    
    fprintf('Batch PCA test (SBR data set): ');
    data = load('tests/SBRDATA.mat');
    tagNames = {'Styrene flow', 'Butadiene flow', 'Feed temperature', ...
                'Reactor temperature', 'Cooling water temperature', ...
                'Reactor jacket tempearture', 'Latex density', ...
                'Conversion', 'Energy released'};
%          
%     assertExceptionThrown('block:invalid_block_type', ...
%                             @block, 'X', {'batch_tag_names', tagNames}, {'nBatches', 52})
%     assertExceptionThrown('block:inconsistent_data_specification', ...
%                             @block, data.X, 'X', 'batch', 'batch_tag_names', tagNames, 'nBatches', 52)
%     assertExceptionThrown('block:number_of_batches_not_specified', ...
%                             @block, data.X, 'X', 'batch' )
%                         
    batch_X = block(data.X, 'X', {'batch_tag_names', tagNames}, {'nBatches', 53});
    options = lvm_opt();     
    options.show_progress = false; 
    options.min_lv = 2;
    batch_PCA = lvm({'X', batch_X},options);
    
    expected = load('tests/SBR-expected.mat');
    
    assertEAE([.17085, .100531], batch_PCA.stats{1}.R2b_a, 5)
    assertEAE(expected.t, batch_PCA.T{1}, 2, true)
    assertEAE(expected.p, batch_PCA.P{1}, 2, true)
    
    fprintf('OK\n');
    
    % TODO(KGD): compare values with the Technometrics paper (55 batches, 10
    % tags, 100 time steps). As on page 45, 46, 47, 48 (table)
return

function PLS_batch_data()    
    fprintf('Batch PLS test (FMC data set): ');

    FMC = load('datasets/FMC.mat');

    % Initial conditions block
    Z = block(FMC.Z);
    Z.add_labels(2, FMC.Znames)   % you can always add labels later on 
    
    % Batch data block (pre-aligned)
    tag_names = {'CTankLvl','DiffPres','DryPress','Power','Torque','Agitator', ...
                 'J-Temp-SP','J-Temp','D-Temp-SP','D-Temp','ClockTime'};

    X = block(FMC.batchSPCData, 'X block',...                       % name of the block
                                {'batch_tag_names', tag_names}, ... % tag names
                                {'batch_names', FMC.Xnames}); 

    % Final quality attributes (FQAs)
    Y = block(FMC.Y, {'col_labels', FMC.Ynames});   % Add labels when creating the block
    
    model = lvm({'Z', Z, 'X', X, 'y', Y}, 2);
    
    fprintf('OK\n');
    
    % TODO(KGD): compare values with the Technometrics paper (55 batches, 10
    % tags, 100 time steps). As on page 45, 46, 47, 48 (table)
return

function PCA_cross_validation_no_missing()
    % Generates several cases of loadings matrices and scores matrices and
    % verifies whether the cross-validation algorithm can calculate the
    % number of components accurately.
    N = 100;
    K = 10;
    A = 5;
    std_T = [2, 1.8, 1.5, 1.2, 1];   % eigenvalues, lambda = std_T .^ 2
    P = create_orthogonal_matrix(K, A);
    T = create_orthogonal_matrix(N, A) .* sqrt(N) .* repmat(std_T, N, 1);
    T = zeros(N, A);
    T(:,1) = create_centered_decreasing_spread(N, 1, std_T(1), 1e-4);
    T(:,1) = T(:,1) ./ std(T(:,1)) .* std_T(1);
    for a = 2:A
        t_a = create_centered_decreasing_spread(N, 1, std_T(a), 1e-4);
        beta_coeff = regress_func(t_a, T(:,a-1), 0);
        resids = t_a - beta_coeff * T(:,a-1);
        resids = resids ./ std(resids) .* std_T(a);
        T(:,a) = resids;
    end
    
    %T = create_centered_decreasing_spread_orthogonal(N, A, 5, 
    X = T*P';
    X = X ./ repmat(std(X),N,1);
    % ??? R2 == [2, 1.8, 1.5, 1.2, 1].^2 ./ sum([2, 1.8, 1.5, 1.2, 1].^2)
    options = lvm_opt(); 
    options.cross_val.use = true;
    
    % This model causes the chi2inv code to crash: "degrees of freedom < 2"
    %PCA_model = lvm({'X', X}, A);
return

function PLS_randomization_tests()
    % Does randomization testing for various datasets
    
    datasets = struct;
    a_dataset = struct;
    a_dataset.name = 'Test name';
    a_dataset.filename = 'tests/';
    a_dataset.expected_A = 0;
    a_dataset.pretreatment = 'MCUV';    
    
    cheese = csvread('tests/cheddar-cheese.csv');
    datasets.cheese = a_dataset;
    datasets.cheese.name = 'Cheddar cheese dataset';
    datasets.cheese.filename = 'tests/cheese.csv';
    datasets.cheese.expected_A = 1;
    datasets.cheese.X = cheese(:,2:4);
    datasets.cheese.Y = cheese(:,5);   
    
    distillation = csvread('tests/distillation-tower.csv');
    datasets.distillation = a_dataset;
    datasets.distillation.name = 'Distillation tower soft sensor';
    datasets.distillation.filename = 'tests/distillation-tower.csv';
    datasets.distillation.expected_A = 7;  % cross-reference with other packages
    datasets.distillation.X = distillation(:,1:end-1);
    datasets.distillation.Y = distillation(:,end);   
        
    datasets.carra = a_dataset;
    datasets.carra.name = 'NIR on carrageenan powders';
    datasets.carra.filename = 'tests/carra.mat';
    datasets.carra.expected_A = 6;
    
    datasets.water = a_dataset;
    datasets.water.name = 'Water content on spruce plugs from Vis-NIR (long test)';
    datasets.water.filename = 'tests/water.mat';
    datasets.water.expected_A = 8;
    
    % This is an interesting dataset: displays moderately high levels of risk
    % in the first 2 components if you MC and scale to UV.
    % Correlation values generally decrease with each component, but this case
    % study shows and increase between components 2 and 3
    %datasets.iso = a_dataset;
    %datasets.iso.name = 'Vis-NIR on spruce with ISO-brightness as Y';
    %datasets.iso.filename = 'tests/ISO_brightness.mat';
    %datasets.iso.expected_A = 5;
    
    datasets.gasoil = a_dataset;
    datasets.gasoil.name = 'Gas-oil NIR predicting hydrogen content';
    datasets.gasoil.filename = 'tests/gasoil.mat';
    datasets.gasoil.expected_A = 8;
    
%     datasets.rgb = a_dataset;
%     datasets.rgb.name = 'Binned histograms from RGB images to predict species';
%     datasets.rgb.filename = 'tests/RGB.mat';
%     datasets.rgb.expected_A = 2;
    
%     datasets.wavelet = a_dataset;
%     datasets.wavelet.name = 'Wavelet coefficients from RGB images';
%     datasets.wavelet.filename = 'tests/wavelet.mat';
%     datasets.wavelet.expected_A = 10;
    
    datasets.internodes = a_dataset;
    datasets.internodes.name = 'MAS/NMR spectra to predict internodes';
    datasets.internodes.filename = 'tests/internodes.mat';
    datasets.internodes.expected_A = 2;
        
    datasets.poplar = a_dataset;
    datasets.poplar.name = 'Poplar classification from NMR';
    datasets.poplar.filename = 'tests/poplar_class.mat';
    datasets.poplar.expected_A = 4;
    
    datasets.kelder = a_dataset;
    datasets.kelder.name = 'Kelder and Greven: QSAR dataset';
    datasets.kelder.filename = 'tests/kelder.mat';
    datasets.kelder.expected_A = 2;
    
%     datasets.hexapep = a_dataset;
%     datasets.hexapep.name = 'Synthesized hexapeptides characterization';
%     datasets.hexapep.filename = 'tests/hexapep.mat';
%     datasets.hexapep.expected_A = 2;
    
    names = fieldnames(datasets);   
    n_tests = numel(names);
    for k = 1:n_tests        
        test = datasets.(names{k});
        fprintf(['Testing randomization: ', test.name])
        try
            data = load(test.filename);
            X = data.Xcal;
            Y = data.Ycal;
        catch ME
            X = test.X;
            Y = test.Y;
        end
        options = lvm_opt(); 
        options.randomize_test.use = true;
        options.show_progress = false;     
        options.randomize_test.show_progress = false;
        if strcmp(test.pretreatment, 'MCUV')
            PLS_model = lvm({'X', X, 'Y', Y}, options);
        end
        assertTrue(PLS_model.A == test.expected_A)
        fprintf(': OK\n');
    end
return

function MBPCA_tests()

    % Does the full multiblock PCA model manually and compares it to the
    % results from the LVM class.

    LDPE = load('tests/LDPE-PLS.mat');
    X1_raw = LDPE.data.blocks{1};

    block_1_vars = [1,2,3,6,8,10,12,14];
    block_2_vars = [4,5,7,9,11,13];
    X_1 = block(X1_raw(:, block_1_vars));
    X_2 = block(X1_raw(:, block_2_vars));
    
    X_raw = {X_1, X_2};    

    % ---------------------------------------------------------------------------
    % Settings for both approaches
    A = 2;

    B = numel(X_raw);
    K_b = zeros(1,B);
    for b = 1:B
        K_b(b) = size(X_raw{b}.data, 2);
    end
    N = size(X_raw{1}.data, 1);

    % Create cell arrays for each block; preprocess each block
    X_mb = cell(1, B);
    for b = 1:B
        X_raw{b} = X_raw{b}.preprocess();
        X_mb{b} = X_raw{b}.data;    
    end

    % ---------------------------------------------------------------------------
    % Merged PCA approach
    % 
    % Afterwards we have to recover the block scores, and block loadings that we 
    % would have otherwise calculated from the full approach.
    T_b_recovered = cell(1, B);
    P_b_recovered = cell(1, B);
    T_s_recovered = zeros(N, A) .* NaN;   % This is identical to T(:,:) from the above PCA
    T_sum_recovered = zeros(N, B, A);     % Superscore array
    P_s_recovered = zeros(B, A) .* NaN;   % Loadings for the superscore array
    for b = 1:B
        T_b_recovered{b} = zeros(N, A);
        P_b_recovered{b} = zeros(K_b(b), A);
    end

    % Statistics are stored here
    stats_PCA = struct();
    stats_PCA.R2X = cell(1, B);
    stats_PCA.R2X_baseline = cell(1,B);
    stats_PCA.R2X_overall = zeros(1,A);

    X_merged = ones(N, sum(K_b)) .* NaN;
    start_col = 1;
    for b = 1:B
        last_col = start_col + K_b(b) - 1;
        X_merged(:, start_col:last_col) = X_mb{b} / sqrt(K_b(b));
        start_col = start_col + K_b(b);

        stats_PCA.R2X_baseline{b} = ssq(X_mb{b} / sqrt(K_b(b)));
    end
    K = size(X_merged, 2);
    assert(abs(sum(cell2mat(stats_PCA.R2X_baseline))-ssq(X_merged)) <sqrt(eps))



    T = zeros(N, A);
    P = zeros(K, A);
    for a = 1:A
        t_a = randn(N, 1);
        t_a_guess = t_a * 2;
        while norm(t_a_guess - t_a) > eps^(2/3)
            t_a_guess = t_a;
            p_a = X_merged' * t_a / (t_a' * t_a);
            p_a = p_a / norm(p_a);
            t_a = X_merged * p_a / (p_a'*p_a);
        end
        T(:,a) = t_a;
        P(:,a) = p_a;

        % Recover the information for each block
        start_col = 1;
        for b = 1:B
            last_col = start_col + K_b(b) - 1;
            % Multiply here by sqrt(K_b(b)) to get the X_portion looking like the
            % X-block that would have come from full multiblock approach.
            X_portion = X_merged(:, start_col:last_col) * sqrt(K_b(b));
            p_b_temp = X_portion' * t_a / (t_a' * t_a);
            p_b_temp = p_b_temp / norm(p_b_temp);
            T_b_recovered{b}(:,a) = X_portion * p_b_temp / (p_b_temp' * p_b_temp) / sqrt(K_b(b));
            T_sum_recovered(:, b, a) = T_b_recovered{b}(:,a);

            P_b_recovered{b}(:,a) = p_b_temp;


            stats_PCA.R2X{b}(a) = ssq(t_a * p_a(start_col:last_col)') / stats_PCA.R2X_baseline{b}; 

            start_col = start_col + K_b(b);    
        end
        P_s_recovered(:,a) = T_sum_recovered(:, :, a)' * t_a / (t_a' * t_a);


        stats_PCA.R2X_overall(a) = ssq(t_a*p_a') / sum(cell2mat(stats_PCA.R2X_baseline));

        % Finally, deflate
        X_merged = X_merged - t_a*p_a';


    end

    % ---------------------------------------------------------------------------
    % Full multiblock approach to calculating scores and loadings for each block, 
    % as well as getting summary scores and loadings (super block).


    % Storage for all variables we want to keep afterwards
    T_sum = zeros(N, B, A); % superblock variables
    T_s = zeros(N, A);      % superblock's scores
    P_s = zeros(B, A);      % superblock's loadings

    % Statistics are stored here
    stats_MB = struct();

    % Block scores and loadings
    T_b = cell(1, B);
    P_b = cell(1, B);
    stats_MB.R2X = cell(1, B);
    stats_MB.R2X_baseline = cell(1,B);
    stats_MB.R2X_overall = zeros(1,A);
    for b = 1:B
        T_b{b} = zeros(N, A);
        P_b{b} = zeros(K_b(b), A);
        stats_MB.R2X{b} = zeros(1, A);
        stats_MB.R2X_baseline{b} = ssq(X_mb{b});
    end

    for a = 1:A

        % The overall consensus superscore from the superblock
        t_a_s = randn(N, 1);
        t_a_s_guess = t_a_s * 2;

        % Block loadings and block scores
        p = cell(B,1);
        t = cell(B,1);    

        while norm(t_a_s_guess - t_a_s) > eps^(9/10)
            % For later, when coming back through the loop
            t_a_s_guess = t_a_s;

            for b = 1:B

                % Loadings for each block
                p{b} = X_mb{b}' * t_a_s / (t_a_s' * t_a_s);
                p{b} = p{b} / norm(p{b});

                % Scores for each block
                t{b} = X_mb{b} * p{b} / (p{b}' * p{b}) / sqrt(K_b(b));

                % Assemble the block scores together to create the super block
                T_sum(:,b,a) = t{b};
            end

            % Regress columns of the super block scores onto the super score summary
            % to calculate the superblock loadings (these describe the importance of
            % each block to the overall concensus score)
            p_a_s = T_sum(:,:,a)' * t_a_s / (t_a_s' * t_a_s);
            p_a_s = p_a_s / norm(p_a_s);

            % Finally, calculate the updated superscore, by regressing rows in the
            % superscore summary block onto the superblock loadings.
            % We leave the denominator here, to emphasize that it will have to be
            % handled when we have missing data.
            t_a_s = T_sum(:,:,a) * p_a_s / (p_a_s' * p_a_s);
        end


        % Deflate each block by using the superscore
        overall_ssq = 0;
        for b = 1:B
            p_deflate = p{b} * p_a_s(b) * sqrt(K_b(b));
            %p_deflate = X_mb{b}' * t_a_s / (t_a_s' * t_a_s);
            % I'm bothered by the fact that we don't normalize p{b} here.
            X_mb{b} = X_mb{b} - t_a_s * p_deflate';

            block_ssq = ssq(t_a_s * p_deflate');
            overall_ssq = overall_ssq + block_ssq;
            stats_MB.R2X{b}(a) = block_ssq / stats_MB.R2X_baseline{b};        
        end
        stats_MB.R2X_overall(a) = overall_ssq/sum(cell2mat(stats_MB.R2X_baseline));

        % Store results for comparison
        T_s(:, a) = t_a_s;
        P_s(:, a) = p_a_s; 
        for b = 1:B
            T_b{b}(:,a) = t{b};
            P_b{b}(:,a) = p{b};
        end

    end

    assertTrue(norm(abs(T) - abs(T_s)) < sqrt(eps))
    assertTrue(norm(abs(T_sum_recovered(:)) - abs(T_sum(:))) < sqrt(eps))
    assertTrue(norm(abs(P_s_recovered) - abs(P_s)) < sqrt(eps))

    for b = 1:B
        assertTrue(norm(abs(T_b_recovered{b}) - abs(T_b{b})) < sqrt(eps))
        assertTrue(norm(abs(P_b_recovered{b}) - abs(P_b{b})) < sqrt(eps))
    end


    % Now predict the scores for each block and the super scores as if were
    % running from scratch

    T_pred_super = zeros(N, A) .* NaN;
    T_pred_block = cell(1, B);
    for b = 1:B
        T_pred_block{b} = zeros(N, A) .* NaN;
    end


    for n = 1:N
        X_new = cell(1, B);
        t_new = cell(1, B);
        t_new_s = zeros(1, A);
        
        for b = 1:B
            X_new{b} = X_raw{b}.data(n,:);
            % Data are already preprocessed
            % X_new{b} = (X_new{b} - X_raw{b}.PP.mean_center) .* X_raw{b}.PP.scaling;
        end
        for a = 1:A
            for b = 1:B
                t_new{b} = X_new{b} * P_b_recovered{b}(:,a) / sqrt(K_b(b));
                T_pred_block{b}(n,a) = t_new{b};
            end
            % Assemble block scores into a single row vector, 1 x B
            T_new_s = cell2mat(t_new);
            % Calculate estimate of superscore, t_new_s
            t_new_s(a) = T_new_s * P_s_recovered(:,a);

            % Deflate X-blocks with this superscore and the block loadings
            % But the block loadings are modified by the superblock's loadings
            for b = 1:B
                X_new{b} = X_new{b} - t_new_s(a) * (P_b_recovered{b}(:,a)' * P_s_recovered(b,a)) * sqrt(K_b(b));
            end
        end
        T_pred_super(n,:) = t_new_s;
    end
    assertTrue(norm(abs(T_pred_super) - abs(T_s)) < sqrt(eps))
    for b = 1:B
        assertTrue(norm(abs(T_pred_block{b}) - abs(T_b_recovered{b})) < sqrt(eps))
    end

    
    % The above code was purely the reference version from which "mbpca" was
    % written.  Now test the mbpca gives the same results
    X_1 = block(X1_raw(:, block_1_vars), 'X1');
    X_2 = block(X1_raw(:, block_2_vars), 'X2');
    
    
    options = lvm_opt();     
    options.show_progress = false; 
    options.min_lv = 2;
    mbmodel = lvm({'X1', X_1, 'X2', X_2}, options);
    
    % Compare block scores and super scores (tolerance levels are different
    % between the models)
    assertEAE(mbmodel.T{1},            T_b{1}, 5, true)
    assertEAE(mbmodel.T{2},            T_b{2}, 5, true)    
    assertEAE(mbmodel.super.T_summary, T_sum,  5, true)
    assertEAE(mbmodel.super.T,         T_s,    5, true)
    
    % Compare superblock's loadings
    assertEAE(mbmodel.super.P,         P_s,    8, true)
    
    % Compare overall R2
    % TODO(KGD): why are these R2 values slightly different?
    % assertEAE(mbmodel.super.stats.R2,  stats_MB.R2X_overall, 5)
    assertEAE(mbmodel.super.stats.R2X,  stats_PCA.R2X_overall, 14)
    
    % Compare block R2 values for block 1 and 2
    assertEAE(mbmodel.stats{1}.R2b_a,  stats_MB.R2X{1}, 5)
    assertEAE(mbmodel.stats{1}.R2b_a,  stats_PCA.R2X{1}, 5)
    assertEAE(mbmodel.stats{2}.R2b_a,  stats_MB.R2X{2}, 5)
    assertEAE(mbmodel.stats{2}.R2b_a,  stats_PCA.R2X{2}, 5)
    
    
    % TODO(KGD): test on new/same data and ensure the apply() function works
return

function MBPLS_tests()

    LDPE = load('tests/LDPE-PLS.mat');
    X_raw = LDPE.data.blocks{1};
    Y = block(LDPE.data.blocks{2});

    block_1_vars = [1,2,3,6,8,10,12,14];
    block_2_vars = [4,5,7,9,11,13];
    X1 = block(X_raw(:, block_1_vars));
    X2 = block(X_raw(:, block_2_vars));

    
    X_raw = {X1, X2};

    % Number of components
    A = 3;

    B = numel(X_raw);
    K_b = zeros(1,B);
    for b = 1:B
        K_b(b) = size(X_raw{b}.data, 2);
    end
    N = size(X_raw{1}.data,1);


    % -----------------------------------------------------------------------
    %
    % Merged PLS approach: merge the X-space blocks together and create model 
    % using the single block PLS mechanism

    % Reset Y-back to the raw data
    Y_data = Y.data;
    level_Y = mean(Y_data);
    spread_Y = std(Y_data);
    Y_data = (Y_data - repmat(level_Y, N, 1)) ./ repmat(spread_Y, N, 1);

    % Create cell arrays for each block; preprocess each block
    X_mb = cell(1, B);
    for b = 1:B
        X_raw{b} = X_raw{b}.preprocess();
        X_mb{b} = X_raw{b}.data;
    end

    % Statistics are stored here
    stats_PLS = struct();
    stats_PLS.R2X = cell(1, B);
    stats_PLS.R2X_baseline = cell(1,B);
    stats_PLS.R2X_overall = zeros(1,A);
    stats_PLS.R2Y_baseline = ssq(Y_data);
    stats_PLS.R2Y_overall = zeros(1,A);
    stats_PLS.SPE = cell(1,B);
    stats_PLS.SPE_overall = zeros(N, A);
    stats_PLS.T2 = cell(1,B);


    X_merged = ones(N, sum(K_b)) .* NaN;
    start_col = 1;
    for b = 1:B
        last_col = start_col + K_b(b) - 1;
        X_merged(:, start_col:last_col) = X_mb{b} / sqrt(K_b(b));
        start_col = start_col + K_b(b);

        stats_PLS.R2X_baseline{b} = ssq(X_mb{b} / sqrt(K_b(b)));
        stats_PLS.SPE{b} = zeros(N, A);
        stats_PLS.T2{b} = zeros(N, A);
    end
    assert(abs(sum(cell2mat(stats_PLS.R2X_baseline))-ssq(X_merged)) <sqrt(eps))
    K = size(X_merged, 2);
    M = size(Y_data, 2);

    T = zeros(N, A);
    U = zeros(N, A);
    P = zeros(K, A);
    W = zeros(K, A);
    C = zeros(M, A);

    % Afterwards we have to recover the block scores, block weights and block 
    % loadings that we would have otherwise calculated from the full approach.
    W_b_recovered = cell(1, B);
    T_b_recovered = cell(1, B);
    P_b_recovered = cell(1, B);
    T_s_recovered = zeros(N, A) .* NaN;
    T_sum_recovered = zeros(N, B, A);
    W_s_recovered = zeros(B, A) .* NaN;
    for b = 1:B
        W_b_recovered{b} = zeros(K_b(b), A);
        T_b_recovered{b} = zeros(N, A);
        P_b_recovered{b} = zeros(K_b(b), A);
    end

    for a = 1:A
        u_a = randn(N, 1);
        u_a_guess = u_a * 2;
        while norm(u_a_guess - u_a) > eps^(6/7)
            u_a_guess = u_a;
            w_a = X_merged' * u_a / (u_a' * u_a);
            w_a = w_a / norm(w_a);
            t_a = X_merged * w_a / (w_a'*w_a);
            c_a = Y_data' * t_a / (t_a' * t_a);
            u_a = Y_data * c_a;%/ (c_a'*c_a);     % <------- divide (optional)     
        end    
        p_a = X_merged' * t_a / (t_a' * t_a);

        T(:,a) = t_a;
        U(:,a) = u_a;
        P(:,a) = p_a;
        W(:,a) = w_a;
        C(:,a) = c_a;
        start_col = 1;
        for b = 1:B
            last_col = start_col + K_b(b) - 1;
            % Multiply here by sqrt(K_b(b)) to get the X_portion looking like the
            % X-block that would have come from full multiblock approach (which
            % only preprocesses each block, but doesn't downweight it).
            X_portion = X_merged(:, start_col:last_col) * sqrt(K_b(b));
            w_b_recovered_temp = X_portion' * u_a / (u_a' * u_a);
            W_b_recovered{b}(:,a) = w_b_recovered_temp / norm(w_b_recovered_temp);
            T_b_recovered{b}(:,a) = X_portion * W_b_recovered{b}(:,a) / (W_b_recovered{b}(:,a)' * W_b_recovered{b}(:,a)) / sqrt(K_b(b));


            T_sum_recovered(:, b, a) = T_b_recovered{b}(:,a);

            % Extract the block loadings now: these are the same block loadings that
            % would have been calculated to deflate each block
            P_b_recovered{b}(:,a) = X_portion' * t_a / (t_a' * t_a);

            stats_PLS.R2X{b}(a) = ssq(t_a * p_a(start_col:last_col)') / stats_PLS.R2X_baseline{b}; 
            start_col = start_col + K_b(b);
        end
        W_s_recovered(:, a) = T_sum_recovered(:, :, a)' * u_a / (u_a' * u_a);
        W_s_recovered(:, a) = W_s_recovered(:, a) / norm(W_s_recovered(:, a));

        % Finally, deflate
        X_merged = X_merged - t_a * p_a';
        Y_data = Y_data - t_a * c_a';

        start_col = 1;

        for b = 1:B
            last_col = start_col + K_b(b) - 1;

            stats_PLS.SPE{b}(:,a) = ssq(X_merged(:,start_col:last_col), 2);
            start_col = start_col + K_b(b);
        end

        stats_PLS.R2X_overall(a) = ssq(t_a * p_a') / sum(cell2mat(stats_PLS.R2X_baseline));
        stats_PLS.R2Y_overall(a) = ssq(t_a * c_a') / stats_PLS.R2Y_baseline;
        stats_PLS.SPE_overall(:,a) = ssq(X_merged, 2);

    end
    Y_hat_pls = T*C';

    %-----------------------------------------------------------------------------
    % Full multiblock implementation to calculate scores and loadings for each 
    % block, as well as getting summary scores and loadings (super block).

    % Reset Y back to its raw version
    Y_data = Y.data;
    level_Y = mean(Y_data);
    spread_Y = std(Y_data);
    Y_data = (Y_data - repmat(level_Y, N, 1)) ./ repmat(spread_Y, N, 1);

    stats_MB = struct();
    stats_MB.R2X = cell(1, B);
    stats_MB.R2X_baseline = cell(1,B);
    stats_MB.R2X_overall = zeros(1,A);
    stats_MB.R2Y_baseline = ssq(Y_data);
    stats_MB.R2Y_overall = zeros(1,A);
    stats_MB.SPE = cell(1,B);

    % Storage for all variables we want to keep afterwards
    T_sum = zeros(N, B, A);
    T_s = zeros(N, A);
    U_s = zeros(N, A);
    W_s = zeros(B, A);
    C_s = zeros(M, A);
    % Block weights, scores and loadings
    W_b = cell(1, B);
    T_b = cell(1, B);
    P_b = cell(1, B);
    for b = 1:B
        W_b{b} = zeros(K_b(b), A);
        T_b{b} = zeros(N, A);
        P_b{b} = zeros(K_b(b), A);
        stats_MB.R2X{b} = zeros(1, A);
        stats_MB.R2X_baseline{b} = ssq(X_mb{b});
        stats_MB.SPE{b} = zeros(N, A);
    end

    for a = 1:A

        % Block loadings and block scores
        w = cell(B,1);
        t = cell(B,1);
        p = cell(B,1);

        % The superscore from the superblock
        u_a = randn(N, 1);
        u_a_guess = u_a * 2;
        while norm(u_a - u_a_guess) > eps^(6/7)
            u_a_guess = u_a;

            % Iterate over all blocks
            for b = 1:B

                % Weights for each block
                w{b} = X_mb{b}' * u_a / (u_a' * u_a);
                w{b} = w{b} / norm(w{b});

                % Block scores
                t{b} = X_mb{b} * w{b} / (w{b}'*w{b}) / sqrt(K_b(b));

                % Assemble block scores into a superblock matrix of scores, T_sum
                T_sum(:,b,a) = t{b};
            end

            % Regress superblock scores columns onto u_a to get superweights    
            w_a_s = T_sum(:,:,a)' * u_a / (u_a' * u_a);
            w_a_s = w_a_s / norm(w_a_s);

            % Regress rows of superscores onto superweights, to get extent of
            % correlation (slope coefficient) with them
            t_a_s = T_sum(:,:,a) * w_a_s / (w_a_s' * w_a_s);

            % Regress columns of Y onto t_a_s, the score vector from the superblock
            c_a_s = Y_data' * t_a_s / (t_a_s' * t_a_s);

            % Finally, calculate the u_a, Y-block score by regressing rows in Y on
            % the Y-block weights, c_a_s
            u_a = Y_data * c_a_s / (c_a_s' * c_a_s);
        end


        % Deflate all X-blocks and Y-block using the superblock's score vector
        overall_X_ssq = 0;
        for b = 1:B
            p{b} = X_mb{b}' * t_a_s / (t_a_s' * t_a_s);
            X_mb{b} = X_mb{b} - t_a_s * p{b}';

            stats_MB.SPE{b}(:,a) = ssq(X_mb{b},2);
            block_ssq = ssq(t_a_s * p{b}');
            overall_X_ssq = overall_X_ssq + block_ssq;
            stats_MB.R2X{b}(a) = block_ssq / stats_MB.R2X_baseline{b};
        end
        Y_data = Y_data - t_a_s * c_a_s';
        stats_MB.R2X_overall(a) = overall_X_ssq / sum(cell2mat(stats_MB.R2X_baseline));
        stats_MB.R2Y_overall(a) = ssq(t_a_s * c_a_s') / stats_MB.R2Y_baseline;

        % Store results for comparison
        W_s(:,a) = w_a_s;
        T_s(:,a) = t_a_s;
        U_s(:,a) = u_a;    
        C_s(:,a) = c_a_s;    
        for b = 1:B
            W_b{b}(:,a) = w{b};
            T_b{b}(:,a) = t{b};
            P_b{b}(:,a) = p{b};
        end

        % Checks 
        stacked_weight = [];
        for b = 1:B
            stacked_weight = [stacked_weight; w{b} * W_s(b,a)];
        end
        assertTrue(norm(abs(stacked_weight) - abs(W(:,a))) < sqrt(eps))

    end
    Y_hat = T_s * C_s';
    

    assertTrue(norm(abs(T) - abs(T_s)) < sqrt(eps))
    assertTrue(norm(abs(C) - abs(C_s)) < sqrt(eps))
    assertTrue(norm(Y_hat - Y_hat_pls) < sqrt(eps))
    assertTrue(norm(abs(T_sum_recovered(:)) - abs(T_sum(:))) < sqrt(eps))
    assertTrue(norm(abs(W_s_recovered) - abs(W_s)) < sqrt(eps))
    for b = 1:B
        assertTrue(norm(abs(W_b_recovered{b}) - abs(W_b{b})) < sqrt(eps))
        assertTrue(norm(abs(T_b_recovered{b}) - abs(T_b{b})) < sqrt(eps))
        assertTrue(norm(abs(P_b_recovered{b}) - abs(P_b{b})) < sqrt(eps))
    end

    % for m = 1:min(M, 4)
    %     subplot(2,2, m)
    %     % Conclusion: little difference, in this example, of block approach vs
    %     % ordinary PLS with all X variables in one block, ito predictions.
    %     %plot(Y_hat_pls_ordinary(:,m)- Y_hat_pls(:,m),'.'), grid
    %     Y_hat_orig = Y_hat_pls_ordinary .* repmat(spread_Y, N, 1) + repmat(level_Y, N, 1);
    %     plot(Y_hat_orig(:,m), Y.data_raw(:,m),'.')
    %     grid on, axis equal
    % end

    % Now predict the Y-values as if they were new data: from scratch
    Y_pred_MBPLS = zeros(N, M);
    for n = 1:N
        X_new = cell(1, B);
        t_new = cell(1, B);
        t_new_s = zeros(1, A);
        for b = 1:B
            % Data are already preprocessed
            X_new{b} = X_raw{b}.data(n,:);
            %X_new{b} = (X_new{b} - X_raw{b}.PP.mean_center) .* X_raw{b}.PP.scaling;
        end
        for a = 1:A
            for b = 1:B
                t_new{b} = X_new{b} * W_b{b}(:,a) / sqrt(K_b(b));
            end
            % Assemble block scores into a single row vector, 1 x B
            T_new_s = cell2mat(t_new);
            % Calculate estimate of superscore, t_new_s
            t_new_s(a) = T_new_s * W_s(:,a);
            % Deflate X-blocks with this superscore and the block loadings
            for b = 1:B
                X_new{b} = X_new{b} - t_new_s(a) * P_b{b}(:,a)';
            end
        end
        Y_pred_MBPLS(n,:) = t_new_s * C_s';
    end
    assertTrue(norm(Y_pred_MBPLS - Y_hat) < sqrt(eps))

    
    % The above code was purely the reference version from which "mbpls" was
    % written.  Now test that mbpls gives the same results
    % ------------------------------------------------------------------------
    X_raw = LDPE.data.blocks{1};
    Y = block(LDPE.data.blocks{2});
    X1 = block(X_raw(:, block_1_vars), 'X1');
    X2 = block(X_raw(:, block_2_vars), 'X2');

   
    options = lvm_opt();     
    options.show_progress = false; 
    options.min_lv = A;
    mbmodel = lvm({'X1', X1, 'X2', X2, 'Y', Y}, options);
    
    % Compare block scores and super scores (tolerance levels are different
    % between the models)
    assertEAE(mbmodel.T{1},            T_b{1}, 6, true)
    assertEAE(mbmodel.T{2},            T_b{2}, 7, true)
    assertEAE(mbmodel.super.T_summary, T_sum,  6, true)
    assertEAE(mbmodel.super.T,         T_s,    6, true)
    
    % Compare superblock's loadings
    assertEAE(mbmodel.super.W,         W_s,    8, true)
    
    % Compare overall R2
    % TODO(KGD): why are these R2 values slightly different?
    % assertEAE(mbmodel.super.stats.R2,  stats_MB.R2X_overall, 5)
    assertEAE(mbmodel.super.stats.R2X,  stats_PLS.R2X_overall, 9)
    assertEAE(mbmodel.super.stats.R2Y,  stats_PLS.R2Y_overall, 8)
    
    % Compare block R2 values for block 1 and 2
    assertEAE(mbmodel.stats{1}.R2b_a,  stats_MB.R2X{1}, 8)
    assertEAE(mbmodel.stats{1}.R2b_a,  stats_PLS.R2X{1},8)
    assertEAE(mbmodel.stats{2}.R2b_a,  stats_MB.R2X{2}, 8)
    assertEAE(mbmodel.stats{2}.R2b_a,  stats_PLS.R2X{2},8)

    % TODO(KGD): compare block loadings, SPE and T2 value, variable-based R2
    % TODO(KGD): test on new/same data and ensure the apply() function works
return

% =========================================================================
% Code from this point onwards is from the MATLAB open-source unit testing 
% suite: http://www.mathworks.com/matlabcentral/fileexchange/22846
%
% Licensed under the BSD license, which means it may be legally
% distributed - see below.  Their code has been modified, where required,
% to make it work with earlier MATLAB versions.
%
%
% Copyright (c) 2010, The MathWorks, Inc.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the The MathWorks, Inc. nor the names 
%       of its contributors may be used to endorse or promote products derived 
%       from this software without specific prior written permission.
%       
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.
% -----------------

function assertTrue(condition)
% Verifies that the ``condition`` is true.  If not, it will throw an
% exception.

%assertTrue Assert that input condition is true
%   assertTrue(CONDITION, MESSAGE) throws an exception containing the string
%   MESSAGE if CONDITION is not true.
%
%   MESSAGE is optional.
%
%   Examples
%   --------
%   % This call returns silently.
%   assertTrue(rand < 1, 'Expected output of rand to be less than 1')
%
%   % This call throws an error.
%   assertTrue(sum(sum(magic(3))) == 0, ...
%       'Expected sum of elements of magic(3) to be 0')
%
%   See also assertEqual, assertFalse

%   Steven L. Eddins
%   Copyright 2008-2010 The MathWorks, Inc.

if nargin < 2
   message = 'Asserted condition is not true.';
end

if ~isscalar(condition) || ~islogical(condition)
   throwAsCaller(MException('assertTrue:invalidCondition', ...
      'CONDITION must be a scalar logical value.'));
end

if ~condition
   throwAsCaller(MException('assertTrue:falseCondition', '%s', message));
end

function assertEAE(A, B, sigfigs, ignore_sign)
% KGD: original code description below.  In this case, sigfigs = number of
% significant figures.

%assertEAE Assert floating-point array elements almost equal.
%   assertEAE(A, B, tol_type, tol, floor_tol) asserts that all
%   elements of floating-point arrays A and B are equal within some tolerance.
%   tol_type can be 'relative' or 'absolute'.  tol and floor_tol are scalar
%   tolerance values.
%
%   If the tolerance type is 'relative', then the tolerance test used is:
%
%       all( abs(A(:) - B(:)) <= tol * max(abs(A(:)), abs(B(:))) + floor_tol )
%
%   If the tolerance type is 'absolute', then the tolerance test used is:
%
%       all( abs(A(:) - B(:)) <= tol )
%
%   tol_type, tol, and floor_tol are all optional.  The default value for
%   tol_type is 'relative'.  If both A and B are double, then the default value
%   for tol and floor_tol is sqrt(eps).  If either A or B is single, then the
%   default value for tol and floor_tol is sqrt(eps('single')).
%
%   If A or B is complex, then the tolerance test is applied independently to
%   the real and imaginary parts.
%
%   Corresponding elements in A and B that are both NaN, or are both infinite
%   with the same sign, are considered to pass the tolerance test.
%
%   assertEAE(A, B, ..., msg) prepends the string msg to the
%   output message if A and B fail the tolerance test.

%   Steven L. Eddins
%   Copyright 2008-2010 The MathWorks, Inc.

if nargin < 4
    ignore_sign = false;
end

if ~isequal(size(A), size(B))
    message = 'Inputs are not the same size.';
    throwAsCaller(MException('assertEAE:sizeMismatch', ...
        '%s', message));
end

if ~(isfloat(A) && isfloat(B))
    message = 'Inputs are not both floating-point.';
    throwAsCaller(MException('assertEAE:notFloat', ...
        '%s', message));
end

if ~isfloat(A) || ~isfloat(B)
    return_early = isequal(A, B);
else
    return_early = false;
end

if ~isequal(size(A), size(B))
    return_early = false;
end

A = A(:);
B = B(:);

reltol = max(100 * max(eps(class(A)), eps(class(B))), 10^(-sigfigs));
if ignore_sign    
    delta = abs(abs(A) - abs(B)) ./ max(max(abs(A), abs(B)), reltol);
else
    delta = abs(A - B) ./ max(max(abs(A), abs(B)), reltol);
end

% Some floating-point values require special handling.
delta((A == 0) & (B == 0)) = 0;
delta(isnan(A) & isnan(B)) = 0;
delta((A == Inf) & (B == Inf)) = 0;
delta((A == -Inf) & (B == -Inf)) = 0;
same = all(delta <= reltol);

if ~same || return_early
    tolerance_message = sprintf('Input elements are not all equal within a relative tolerance of %g', reltol);
    throwAsCaller(MException('assertEAE:tolExceeded', ...
        '%s', tolerance_message));
end

function assertExceptionThrown(expectedId, f, varargin)
% KGD: I have changed the function signature to be compatible with Python.

%assertExceptionThrown Assert that specified exception is thrown
%   assertExceptionThrown(F, expectedId) calls the function handle F with no
%   input arguments.  If the result is a thrown exception whose identifier is
%   expectedId, then assertExceptionThrown returns silently.  If no exception is
%   thrown, then assertExceptionThrown throws an exception with identifier equal
%   to 'assertExceptionThrown:noException'.  If a different exception is thrown,
%   then assertExceptionThrown throws an exception identifier equal to
%   'assertExceptionThrown:wrongException'.
%
%   assertExceptionThrown(F, expectedId, msg) prepends the string msg to the
%   assertion message.
%
%   Example
%   -------
%   % This call returns silently.
%   f = @() error('a:b:c', 'error message');
%   assertExceptionThrown(f, 'a:b:c');
%
%   % This call returns silently.
%   assertExceptionThrown(@() sin, 'MATLAB:minrhs');
%
%   % This call throws an error because calling sin(pi) does not error.
%   assertExceptionThrown(@() sin(pi), 'MATLAB:foo');

%   Steven L. Eddins
%   Copyright 2008-2010 The MathWorks, Inc.

noException = false;
try
    f(varargin{:});
    noException = true;
    
catch exception
    if ~strcmp(exception.identifier, expectedId)
        message = sprintf('Expected exception %s but got exception %s.', ...
            expectedId, exception.identifier);
        throwAsCaller(MException('assertExceptionThrown:wrongException', ...
            '%s', message));
    end
end

if noException
    message = sprintf('Expected exception "%s", but none thrown.', ...
        expectedId);
    throwAsCaller(MException('assertExceptionThrown:noException', '%s', message));
end

