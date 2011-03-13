function tests(varargin)
    close all;
    test_significant_figures()
    Wold_article_PCA_test()
    basic_PLS_test()
    PCA_no_missing_data()  
    PLS_no_missing_data()
    PCA_with_missing_data()
    PLS_with_missing_data()
    
    PCA_batch_data()
    PCA_cross_validation_no_missing()
    
    PLS_randomization_tests()
    
    
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
    assertElementsAlmostEqual(0.5412, 0.5414, 3)
    assertExceptionThrown('assertElementsAlmostEqual:tolExceeded', ...
                            @assertElementsAlmostEqual, 0.5412, 0.5414, 4)
    assertElementsAlmostEqual(1.5412E-5, 1.5414E-5, 4)
    assertExceptionThrown('assertElementsAlmostEqual:tolExceeded', ...
                @assertElementsAlmostEqual, 1.5412E-5, 1.5414E-5, 5)
    %1.5412 == 1.5414       is True if sig_figs = 4, but False if sig_figs = 5
    %1.5412E-5 == 1.5414E-5 is True if sig_figs = 4, but False if sig_figs = 5
    %1.5412E+5 == 1.5414E+5 is True if sig_figs = 4, but False if sig_figs = 5
return

function Wold_article_PCA_test()
%   Tests from the PCA paper by Wold, Esbensen and Geladi, 1987
%   Principal Component Analysis, Chemometrics and Intelligent Laboratory
%   Systems, v 2, p37-52; http://dx.doi.org/10.1016/0169-7439(87)80084-9

    X_raw = [3, 4, 2, 2; 4, 3, 4, 3; 5.0, 5, 6, 4];
    X = block(X_raw);
    PCA_model_1 = lvm({'X', X}, 1);

    X = block(X_raw);
    PCA_model_2 = lvm({'X', X}, 2);

    % The mean centering vector should be [4, 4, 4, 3], page 40
    assertTrue(all(PCA_model_2.PP{1}.mean_center == [4, 4, 4, 3]));

    % The (inverted) scaling vector [1, 1, 0.5, 1], page 40
    assertTrue(all(PCA_model_2.PP{2}.scaling == [1, 1, 0.5, 1]));
    
    % With 2 components, the loadings are, page 40
    %  P.T = [ 0.5410, 0.3493,  0.5410,  0.5410],
    %        [-0.2017, 0.9370, -0.2017, -0.2017]])
    P = PCA_model_2.P{1};
    assertElementsAlmostEqual(P(:,1), [0.5410, 0.3493, 0.5410, 0.5410]', 2);
    assertElementsAlmostEqual(P(:,2), [-0.2017, 0.9370, -0.2017, -0.2017]', 2);

    T = PCA_model_2.T{1};
    assertElementsAlmostEqual(T(:,1), [-1.6229, -0.3493, 1.9723]', 2)
    assertElementsAlmostEqual(T(:,2), [0.6051, -0.9370, 0.3319]', 2)
    
    Tests for VIP, T2 and SPE

    % R2 values, given on page 43
    R2_a = PCA_model_2.stats{1}.R2_a;
    assertElementsAlmostEqual(R2_a, [0.831; 0.169], 2)

    % SS values, on page 43
    SS_X = ssq(PCA_model_2.blocks{1}.data, 1);
    assertElementsAlmostEqual(SS_X, [0.0, 0.0, 0.0, 0.0], 3)

    % The remaining sum of squares, on page 43
    SS_X = ssq(PCA_model_1.blocks{1}.data, 1);
    assertElementsAlmostEqual(SS_X, [0.0551, 1.189, 0.0551, 0.0551], 3)

    % Testing data.  2 rows of new observations.
    X_test_raw = [3, 4, 3, 4; 1, 2, 3, 4.0];
    X_test = block(X_test_raw);
    assertElementsAlmostEqual(X_test.data, [3, 4, 3, 4; 1, 2, 3, 4.0],5);
    testing_type_A = PCA_model_1.apply({'X', X_test});      % send in a block variable
    testing_type_B = PCA_model_1.apply({'X', X_test_raw});  % send in a raw array 
    assertElementsAlmostEqual(testing_type_A.T{1}, [-0.2705, -2.0511]', 4)
    assertElementsAlmostEqual(testing_type_B.T{1}, [-0.2705, -2.0511]', 4)
    
    testing_type_C = PCA_model_2.apply({'X', X_test});      % send in a block variable
    testing_type_D = PCA_model_2.apply({'X', X_test_raw});  % send in a raw array 
    assertElementsAlmostEqual(testing_type_C.T{1}, [-0.2705, -2.0511; 0.1009, -1.3698]', 3)
    assertElementsAlmostEqual(testing_type_D.T{1}, [-0.2705, -2.0511; 0.1009, -1.3698]', 3)
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
    assertElementsAlmostEqual(X.data, raw_data.scaled_blocks{1}, 4)
    
    % Build the PCA model with LVM.m
    % ------------------------------
    A = exp_m.A;
    PCA = lvm({'X', X}, A);
    
    
    % Now test the PCA model
    % ----------------------
    % T-scores
    scores_col = 8:9;
    T = exp_m.observations{1}.data(:, scores_col);
    assertElementsAlmostEqual(PCA.blocks{1}.T, T, 2);
    
    % T^2
    T2_col = 2;
    T2 = exp_m.observations{1}.data(:, T2_col);
    assertElementsAlmostEqual(PCA.blocks{1}.stats.T2(:,PCA.A), T2, 4);
    
    % X-hat
    X_hat_col = 3:7;
    X_hat = exp_m.observations{1}.data(:, X_hat_col);
    X_hat_calc = PCA.blocks{1}.T * PCA.blocks{1}.P';
    assertElementsAlmostEqual(X_hat_calc, X_hat, 2);
    
      % Loadings
    loadings_col = 1:2;
    P = exp_m.variables{1}.data(:, loadings_col);
    assertElementsAlmostEqual(PCA.blocks{1}.P, P, 4);
    
    % R2-per variable(k)-per component(a)
    R2_col = 4:5;
    R2k_a = exp_m.variables{1}.data(:, R2_col);
    assertElementsAlmostEqual(PCA.blocks{1}.stats.R2k_a, R2k_a, 4);  
    
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
    assertElementsAlmostEqual(X.data, raw_data.scaled_blocks{1}, 2)
    
    Y = block(raw_data.blocks{2});
    Y = Y.preprocess();
    assertElementsAlmostEqual(Y.data, raw_data.scaled_blocks{2}, 4)
    
    % Build the PCA model with LVM.m
    % ------------------------------
    A = exp_m.A;
    PLS = lvm({'X', X, 'Y', Y}, A);


    % Now test the PLS model
    % ----------------------
    % T-scores
    scores_col = 3:8;
    T = exp_m.observations{1}.data(:, scores_col);
    assertElementsAlmostEqual(PLS.blocks{1}.T, T, 2, true);
    
    % T^2
    T2_col = 2;
    T2 = exp_m.observations{1}.data(:, T2_col);
    assertElementsAlmostEqual(PLS.blocks{1}.stats.T2(:,PLS.A), T2, 4);
    
    % Y-hat
    Y_hat_col = 7:11;
    Y_hat = exp_m.observations{2}.data(:, Y_hat_col);
    Y_hat_calc = PLS.blocks{2}.data_pred;
    Y_hat_PP = PLS.blocks{2}.preprocess(block(Y_hat_calc));
    assertElementsAlmostEqual(Y_hat_PP.data, Y_hat, 2);
    
    % W-Loadings
    loadings_col = 1:6;
    W = exp_m.variables{1}.data(:, loadings_col);
    assertElementsAlmostEqual(PLS.blocks{1}.W, W, 3, true);
    
    % R2-per variable(k)-cumulative: X-space
    R2X_col = 24;
    R2kX_cum = exp_m.variables{1}.data(:, R2X_col);
    assertElementsAlmostEqual(PLS.blocks{1}.stats.R2k_cum(:,end), R2kX_cum, 4); 
    
    % R2-per variable(k)-per component(a): Y-space
    R2Y_col = 8;
    R2kY_cum = exp_m.variables{2}.data(:, R2Y_col);
    assertElementsAlmostEqual(PLS.blocks{2}.stats.R2k_cum(:,end), R2kY_cum, 4); 
    
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
    PCA = lvm({'X', X}, A);
    
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
         
    assertExceptionThrown('block:invalid_block_type', ...
                            @block, 'X', 'batch', 'tagNames', tagNames, 'nBatches', 52)        
    assertExceptionThrown('block:inconsistent_data_specification', ...
                            @block, data.X, 'X', 'batch', 'tagNames', tagNames, 'nBatches', 52)
    assertExceptionThrown('block:number_of_batches_not_specified', ...
                            @block, data.X, 'X', 'batch' )
    batch_X = block(data.X, 'X', 'batch', 'tagNames', tagNames, 'nBatches', 53);
    options = lvm_opt();     
    options.show_progress = false; 
    options.min_lv = 2;
    batch_PCA = lvm({'X', batch_X},options);
    
    batchspc_data = load('tests/SBR-batchspc.mat');
    
    assertElementsAlmostEqual([.17085, .100531]', batch_PCA.X.stats.R2_a, 5)
    assertElementsAlmostEqual(batchspc_data.t, batch_PCA.X.T, 2, true)
    assertElementsAlmostEqual(batchspc_data.p, batch_PCA.X.P, 2, true)
    
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
    PCA_model = lvm({'X', X}, A);
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

function assertElementsAlmostEqual(A, B, sigfigs, ignore_sign)
% KGD: original code description below.  In this case, sigfigs = number of
% significant figures.

%assertElementsAlmostEqual Assert floating-point array elements almost equal.
%   assertElementsAlmostEqual(A, B, tol_type, tol, floor_tol) asserts that all
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
%   assertElementsAlmostEqual(A, B, ..., msg) prepends the string msg to the
%   output message if A and B fail the tolerance test.

%   Steven L. Eddins
%   Copyright 2008-2010 The MathWorks, Inc.

if nargin < 4
    ignore_sign = false;
end

if ~isequal(size(A), size(B))
    message = 'Inputs are not the same size.';
    throwAsCaller(MException('assertElementsAlmostEqual:sizeMismatch', ...
        '%s', message));
end

if ~(isfloat(A) && isfloat(B))
    message = 'Inputs are not both floating-point.';
    throwAsCaller(MException('assertElementsAlmostEqual:notFloat', ...
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
    throwAsCaller(MException('assertElementsAlmostEqual:tolExceeded', ...
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
