classdef lvm 
    % Class attributes
    properties
        model_type = '';
        blocks = {};  % a cell array of data blocks
        superB = [];  % super-block
        combnd = [];  % combined X-blocks (used to fit multiblock models)
        
        A = 0; % number of latent variables
        B = 0; % number of blocks
        opt = struct();    % Model options
        stats = struct('timing', [], 'itern', []);  % Model statistics (time, iterations)
    end
    
    methods
        
        
        
        function self = calc_model(self, A)
            
           
            % Single-block X and single-block Y, PLS model
            elseif strcmp(self.model_type, 'npls') && self.B == 2
                [self, self.blocks{1}, self.blocks{2}] = fit_PLS(self, self.blocks{1}, self.blocks{2}, A);

            % Single-block X and single-block Y, PLS model
            elseif strcmp(self.model_type, 'mbpca')
                error('lvm:mbpca', 'This model type is not available yet')
                
            % Multi-block X and single-block Y, MBPLS model
            elseif strcmp(self.model_type, 'mbpls')
                
                % 1. Block scale each of the "X-space" blocks
                K = cell(self.B-1, 1);
                K_total = 0;
                ssq_X_blocks = cell(self.B-1,1);
                for b = 1:self.B-1
                    if self.opt.mbpls.block_scale_X
                        K{b} = self.blocks{b}.K;
                        K_total = K_total + K{b};
                        ssq_X_blocks{b} = ssq(self.blocks{b}.data, 1);
                        self.blocks{b}.data = self.blocks{b}.data ./ sqrt(K{b});
                    end
                end
                
                % Then combine the X-blocks into a new single X-block:
                N = self.blocks{end}.N;
                bigX = zeros(N, K_total);
                start_col = 1;
                end_col = 0;
                for b = 1:self.B-1
                    end_col = end_col + K{b};
                    K{b} = start_col:end_col;
                    bigX(:, K{b}) = self.blocks{b}.data;
                    start_col = start_col + K{b}(end);                    
                end
                self.combnd = block(bigX);
                self.combnd.is_preprocessed = true;
                
                [self, self.combnd, self.blocks{end}] = fit_PLS(self, self.combnd, self.blocks{end}, A);
                
                % Now assign weights, scores and so on to each of the blocks.
                % Also place the R2 and other stats into each block
                
                
                % Calculate block variables:                
                w_MBPLS = cell(1, self.B-1);
                p_MBPLS = cell(1, self.B-1);
                r_MBPLS = cell(1, self.B-1);
                t_MBPLS = cell(1, self.B-1);
                X_super_MBPLS = zeros(N, self.B-1, A);
                w_super_MBPLS = zeros(self.B-1, A);
                T_super_MBPLS = self.combnd.T;
                R2_X = cell(1, self.B-1);
                for b = 1:self.B-1
                    w_MBPLS{b} = zeros(numel(K{b}), A);                    
                    p_MBPLS{b} = zeros(numel(K{b}), A);
                    r_MBPLS{b} = zeros(numel(K{b}), A);
                    R2_X{b} = zeros(numel(K{b}), A);
                    t_MBPLS{b} = zeros(N, A);
                    
                    self.blocks{b}.stats.start_SS_col = ssq_X_blocks{b};
                    temp = self.blocks{b}.data .* sqrt(numel(K{b}));
                    for a = 1:A                        
                        %w_MBPLS{b}(:,a) = regress_func(temp, self.Y.U(:,a), self.blocks{b}.has_missing);
                        %w_MBPLS{b}(:,a) = w_MBPLS{b}(:,a)/norm(w_MBPLS{b}(:,a));
                        %w_MBPLS{b}(:,a) = self.combnd.W(K{b}, a);
                        %w_MBPLS{b}(:,a) = w_MBPLS{b}(:,a) / norm(w_MBPLS{b}(:,a));
                        r_MBPLS{b}(:,a) = self.combnd.R(K{b}, a);
                        r_MBPLS{b}(:,a) = r_MBPLS{b}(:,a) / norm(r_MBPLS{b}(:,a));
                        t_MBPLS{b}(:,a) = regress_func(temp, r_MBPLS{b}(:,a), self.blocks{b}.has_missing);
                        t_MBPLS{b}(:,a) = t_MBPLS{b}(:,a) / sqrt(numel(K{b}));
                        p_MBPLS{b}(:,a) = self.combnd.P(K{b}, a);
                        temp = temp - T_super_MBPLS(:,a) * p_MBPLS{b}(:,a)';
                        X_super_MBPLS(:, b, a) = t_MBPLS{b}(:,a);
                        self.blocks{b}.stats.SS_col(:,a) = ssq(temp,1)';
                        self.blocks{b}.stats.R2k_cum(:,a) = self.blocks{b}.stats.SS_col(:,a) ./ ssq_X_blocks{b}';
                        %sum(self.blocks{b}.stats.SS_col(:,a)) / sum(ssq_X_blocks{b})                         
                    end
                end
                for a = 1:A
                    w_super_MBPLS(:,a) = X_super_MBPLS(:,:,a)' * self.Y.U(:,a) / (self.Y.U(:,a)'*self.Y.U(:,a));
                    w_super_MBPLS(:,a) = w_super_MBPLS(:,a) / norm(w_super_MBPLS(:,a));
                end
                
                self.superB = block([]);
                
            end
        end % ``calc_model``
        
        function state = apply(self, other, varargin)
            % Apply the existing model to a ``other`` data set.
            % This function is publically available:
            %
            % > X_build = [3, 4, 2, 2; 4, 3, 4, 3; 5.0, 5, 6, 4];
            % > PCA_model = lvm({'X', X_build}, 2)
            % > X_test = [3, 4, 3, 4; 1, 2, 3, 4.0];
            % > pred = PCA_model.apply({'X', X_test})    # uses all components
            % > pred = PCA_model.apply({'X', X_test}, 1) # uses only 1 component 

            
            % ``varargin`` must by given for batch datasets: carries state
            % information that is used while processing a single batch.
            
            if nargin==3 && isa(varargin{1}, 'struct')
                state = varargin{1};
                state.j = state.j + 1;
            else
                % Initialize storage vectors
                state = struct;
                
                % Store the new data blocks
                state.blocks = lvm_setup_from_data(other);
                state.j = 1;
                state.A = self.A;
            end
            if nargin==3 && isa(varargin{1}, 'numeric')
                state.A = varargin{1};
            end
            
            % Resize the storage for ``A`` components
            for b = 1:self.B
                try
                    new_block = state.blocks{b};
                catch ME
                    % When using a model on new data we may not always have
                    % all the blocks present.  Fill them in with empty, new
                    % blocks.
                    state.blocks{b} = block();
                    new_block = state.blocks{b};
                end
                if ~state.blocks{b}.is_preprocessed
                    state.blocks{b} = self.blocks{b}.preprocess(new_block);
                end
                state.blocks{b} = state.blocks{b}.initialize_storage(state.A);
            end
            
            % Single-block PCA model (non-batch)
            if strcmp(self.model_type, 'pca') && self.B == 1
                state.blocks{1} = apply_PCA(self, state.blocks{1}, state.A);
                
            % Single-block X and single-block Y, PLS model
            elseif strcmp(self.model_type, 'npls') && self.B == 2
                % Single X-block and single Y-block
                % Predictions are located in state.blocks{2}, the
                % predicted Y-block.
                [state.blocks{1}, state.blocks{2}] = apply_PLS(self, state.blocks{1}, state.blocks{2}, state.A);
            end
            state = calc_statistics(self, state);  
        end % ``apply_model``
        
        function self = monitoring_limits(self)
            % Calculates the monitoring limits for a batch model.
            % This of course assumes that ``self`` only contains the good
            % reference data from which to build the limits.
            
            batch_blocks = false(1, self.B);
            for b = 1:self.B                
                if strcmpi(self.blocks{b}.block_type, 'batch')
                    batch_blocks(b) = true;
                end
            end
            batch_blocks = find(batch_blocks);
            
            for b = batch_blocks
                % Iterate over every batch.  Allocate space for calculated
                % results.
                batch = self.blocks{b};
                batch.T_j = zeros(batch.N, self.A, batch.J);
                batch.error_j = zeros(batch.N, batch.K);
                
                show_progress = self.opt.batch.monitoring_limits_show_progress;
                show_progress = self.opt.show_progress;
                if show_progress
                    h = awaitbar(0, sprintf('Calculating monitoring limits for block %s', batch.name));
                end
                stop_early = false;
                SPE_j_temp = zeros(batch.N, batch.J);
                for n = 1:batch.N
                    if show_progress
                        perc = floor(n/batch.N*100);
                        stop_early = awaitbar(perc/100,h,sprintf('Processing batches for limits %d. [%d%%]',n, perc));
                        if stop_early                             
                            break; 
                        end	
                    end
                    % Reset the vector for the current batch
                    batch.lim.x_pp = block(ones(1, batch.nTags * batch.J) .* NaN);
                    
                    % Extract data from the cell array, one cell per batch
                    test_data = batch.data_raw{n};
                    
                    for j = 1:batch.J
                        idx_beg = batch.nTags*(j-1)+1;
                        idx_end = batch.nTags*j;
                        
                        
                        % Preprocess the new data (center and scale it)
                        % We rely on MATLAB's ability to handle NaN's: entries that were NaN
                        % in "test_data" and still NaN's in "x_pp"
                        x_pp = test_data(j,:) - batch.PP.mean_center(1,idx_beg:idx_end);
                        x_pp = x_pp .* batch.PP.scaling(1,idx_beg:idx_end);
                        
                        % Store the preprocessed vector back in the array
                        batch.lim.x_pp.data(idx_beg:idx_end) = x_pp;
                        
                        
                        % Apply the model to these new data
                        if strcmp(self.model_type, 'pca')
                            out = self.apply_PCA(batch.lim.x_pp, self.A, 'quick');
                        elseif strcmp(self.model_type, 'npls')
                            out = self.apply_PLS(batch.lim.x_pp, [], self.A, 'quick');
                            batch.data_pred_pp = out.T * batch.C';
                            batch.data_pred = batch.data_pred_pp / batch.PP.scaling + batch.PP.mean_center;
                        end
                        batch.T_j(n, :, j) = out.T;
                        batch.error_j(n, idx_beg:idx_end) = out.data(1, idx_beg:idx_end);
                        SPE_j_temp(n,j) = ssq(out.data(1, idx_beg:idx_end));
                        %batch.stats.SPE_j(n, j) = out.stats.SPE(1, end); % use only the last component
                        
                    end % ``j=1, 2, ... J``
                end % ``n=1, 2, ... N`` 
                
                if not(stop_early)
                    alpha = 0.95;
                    
                    
                    std_t_j = squeeze(std(batch.T_j,0, 1))';  % J by A matrix
                    t_crit = abs(my_tinv((1-alpha)/2, batch.N-1));
                    % From the central limit theorem, assuming each batch is 
                    % independent from the next (because we are calculating
                    % out std_t_j over the batch dimension, not the time
                    % dimension).  Clearly each time step is not independent.
                    % We inflate the limits slightly: see Nomikos and
                    % MacGregor's article in Technometrics
                    batch.lim.t_j = t_crit * std_t_j * (1 + 1/batch.N);
                    
                    % Variance/covariance of the score matrix over time.
                    % The scores, especially at the start of the batch are
                    % actually correlated; only at the very end of the
                    % batch do they become uncorrelated
                    for j = 1:batch.J
                        scores = batch.T_j(:,:,j);
                        sigma_T = inv(scores'*scores/(batch.N-1));
                        for n = 1:batch.N
                            batch.stats.T2_j(n, j) = batch.T_j(n,:,j) * sigma_T * batch.T_j(n,:,j)';
                        end
                        
                    end
                    % Calculate the T2 limit
                    % Strange that you can calculate it without reference to
                    % any of the batch data.  Also, I would have expected it
                    % to vary during the batch, because t1 and t2 limits are
                    % large initially.
                    N = batch.N;
                    for a = 1:self.A
                        mult = a*(N-1)*(N+1)/(N*(N-a));
                        limit = my_finv(alpha, a, N-(self.A));
                        batch.lim.T2_j(:,a) = mult * limit;
                        % This value should agree with batch.lim.T2(:,a)
                        % TODO(KGD): slight discrepancy in batch SBR dataset
                    end
                    
                    % Calculate SPE time-varying limits
                    % Our SPE definition = sqrt(e'e / K), where K = number of
                    % entries in vector ``e``.  In the SPE_j case that is the
                    % number of tags.
                    % Apply smoothing window here: see Nomikos thesis, p 66.
                    % ``w`` should be a function of how "jumpy" the SPE plot
                    % looks.  Right now, I'm going to set ``w`` proportional
                    % to the number of time samples in a batch
                    w = max(1, ceil(0.012/2 * batch.J));
                    for j = 1:batch.J
                        start_idx = max(1, j-w);
                        end_idx = min(batch.J, j+w);
                        SPE_values = SPE_j_temp(:, start_idx:end_idx);
                        batch.lim.SPE_j(j) = sqrt(calculate_SPE_limit(SPE_values, alpha)/batch.nTags);
                    end
                    SPE_j_temp = SPE_j_temp ./ batch.nTags;
                    batch.stats.SPE_j = sqrt(SPE_j_temp);
                end
                % Write update information back again
                self.blocks{b} = batch;
                
            end % ``for every batch block``
        end
                  
        
        function self = calc_limits(self)
            
        end %``calc_limits``
        
        
        
        
        
        function block = apply_PCA(self, block, A, varargin)
            % Applies a PCA model to the given ``block`` of (new) data.
            % TODO(KGD): there is much similarity between this function and
            % the fit_PCA model function.  Extract common parts!
            
            % Used by internal functions to get the same output, but without
            % various statistical calculations, like R2, etc
            quick = false;
            if nargin == 4
                if strcmp(varargin{1}, 'quick')
                    quick = true;
                end
            end            
            
            tolerance = self.opt.tolerance;
            X = block.data;
            [N, K] = size(X);            
            
            if quick
                % I know the code is duplicated to that below, but that's to
                % avoid a whole lot of "if quick, .... end" mess.
                for a = 1 : min(self.A, A)

                    % Regress each row in X on the p_a vector, and store the
                    % regression coefficient in t_a
                    t_a = regress_func(X, self.blocks{1}.P(:,a), block.has_missing);

                    % Deflate the X-matrix
                    X = X - t_a * self.blocks{1}.P(:,a)';
                    row_SSX = ssq(X, 2);
                    block.stats.SPE(:,a) = sqrt(row_SSX/K);
                    block.T(:,a) = t_a;
                end % looping on ``a`` latent variables
                block.data = X;
                block.A = a;
                return
            end % ``quick`` method
            
            which_components = 1 : min(self.A, A);
            
            % TODO(KGD): have a check here to verify the size of new data
            % is compatible with ``self``.

            % Baseline for all R^2 calculations
            
            for a = which_components
                tic;
                if a == 1                
                    block.stats.start_SS_col = ssq(X, 1);
                    initial_ssq = block.stats.start_SS_col;
                else
                    initial_ssq = ssq(X, 1);
                end
                % Baseline for all R^2 calculations
                start_SS_col = block.stats.start_SS_col;
                if all(initial_ssq < tolerance)
                        warning('lvm:apply_PCA', 'There is no variance left in the data')
                end                
                
                % Regress each row in X on the p_a vector, and store the
                % regression coefficient in t_a
                t_a = regress_func(X, self.blocks{1}.P(:,a), block.has_missing);

                % Deflate the X-matrix
                X = X - t_a * self.blocks{1}.P(:,a)';
                % These are the Residual Sums of Squares (RSS); i.e X-X_hat
                row_SSX = ssq(X, 2);
                col_SSX = ssq(X, 1);

                block.stats.SPE(:,a) = sqrt(row_SSX/K);
                block.stats.deflated_SS_col(:,a) = col_SSX(:);
                block.stats.R2k_cum(:,a) = 1 - col_SSX./start_SS_col;

                % Cumulative R2 value for the whole block
                block.stats.R2(a) = 1 - sum(row_SSX)/sum(start_SS_col);

                % Store results
                block.T(:,a) = t_a;
                block.A = a;
                self.stats.timing(a) = toc;
            end % looping on ``a`` latent variables
            
            block.data = X; % Write the deflated array back to the block
            
        end % ``apply_PCA``
        
        function [self, blockX, blockY] = fit_PLS(self, blockX, blockY, A)
            % Fits a PLS latent variable model between ``blockX`` and ``blockY``.
            %  
            % 1.  Höskuldsson, PLS regression methods, Journal of Chemometrics,
            %     2(3), 211-228, 1998, http://dx.doi.org/10.1002/cem.1180020306

            tolerance = self.opt.tolerance;

            X = blockX.data;
            Y = blockY.data;
            which_components = max(blockY.A+1, 1) : A;
            [N, K] = size(X);
            Ny = size(Y, 1);
            if N ~= Ny
                warning('lvm:fit_PLS', 'The number of observations in both blocks must match.')
            end
            
            if blockY.has_missing
                if any(sum(blockY.mmap, 2) == 0)
                    warning('lvm:fit_PLS', ...
                            ['Cannot handle the case yet where the entire '...
                             'observation in Y-matrix is missing.  Please '...
                             'remove those rows and refit model'])
                end
            end

            for a = which_components
                tic;
                itern = 0;
                if not(self.opt.randomize_test.quick_return)
                    if a == 1
                        % Baseline for all R^2 calcs
                        initial_ssq_X = ssq(X, 1);
                        initial_ssq_Y = ssq(Y, 1);       
                        blockX.stats.start_SS_col = initial_ssq_X;
                        blockY.stats.start_SS_col = initial_ssq_Y;
                    else
                        initial_ssq_X = ssq(X, 1);
                        initial_ssq_Y = ssq(Y, 1);
                    end
                    
                    if all(initial_ssq_X < tolerance)
                        warning('lvm:fit_PLS', 'There is no variance left in the X-data. Returning.');
                        return
                    end
                    if all(initial_ssq_Y < tolerance)
                        warning('lvm:fit_PLS', 'There is no variance left in the Y-data. Returning.');
                        return
                    end
                end
                
                % Initialize u_a with random numbers, or carefully select 
                % a column from Y                
                rand('state', 0);
                u_a_guess = rand(N,1)*2-1;
                u_a = u_a_guess + 1.0;

                while not(should_terminate(u_a_guess, u_a, itern, tolerance, self))

                    % 0: starting point for convergence checking on next loop
                    u_a_guess = u_a;

                    % 1: Regress the score, u_a, onto every column in X, compute the
                    %    regression coefficient and store in w_a
                    % w_a = X.T * u_a / (u_a.T * u_a)
                    w_a = regress_func(X, u_a, blockX.has_missing);

                    % 2: Normalize w_a to unit length
                    w_a = w_a / norm(w_a);
                    r_a = w_a;
                    
                    % TODO(KGD): part of this can be taken out of the while loop
                    if strcmp(self.model_type, 'mbpls') && not(self.opt.mbpls.deflate_X)
                        r_a = eye(K);
                        for local_a = 2:a
                           r_a = r_a * (eye(K) - blockX.W(:,local_a-1) * blockX.P(:,local_a-1)');
                        end
                        r_a = r_a * w_a;
                        t_a = regress_func(X, r_a, blockX.has_missing);
                    else
                        % 3: Now regress each row in X on the w_a vector, and store the
                        %    regression coefficient in t_a
                        % t_a = X * w_a / (w_a.T * w_a)
                        t_a = regress_func(X, w_a, blockX.has_missing);
                    end
                    
                    % 4: Now regress score, t_a, onto every column in Y, compute the
                    %    regression coefficient and store in c_a
                    % c_a = Y.T * t_a / (t_a.T * t_a)
                    c_a = regress_func(Y, t_a, blockY.has_missing);

                    % 5: Now regress each row in Y on the c_a vector, and store the
                    %    regression coefficient in u_a
                    % u_a = Y * c_a / (c_a.T * c_a)
                    %
                    % TODO(KGD):  % Still handle case when entire row in Y is missing
                    u_a = regress_func(Y, c_a, blockY.has_missing);

                    itern = itern + 1;
                end
                self.stats.itern(a) = itern; 
                
                % 6: To deflate the X-matrix we need to calculate the
                % loadings for the X-space.  Regress columns of t_a onto each
                % column in X and calculate loadings, p_a.  Use this p_a to
                % deflate afterwards.
                % Note the similarity with step 4! and that similarity helps
                % understand the deflation process.
                p_a = regress_func(X, t_a, blockX.has_missing); 

                
                % Store results
                % -------------
                % Flip the signs of the column vectors in P so that the largest
                % magnitude element is positive (Wold, Esbensen, Geladi, PCA,
                % CILS, 1987, p 42)
                [max_el, max_el_idx] = max(abs(p_a));                
                if sign(p_a(max_el_idx)) < 1
                    blockX.W(:,a) = -1.0 * w_a;
                    blockX.P(:,a) = -1.0 * p_a;
                    blockX.T(:,a) = -1.0 * t_a;
                    blockX.R(:,a) = -1.0 * r_a;
                    blockY.C(:,a) = -1.0 * c_a;
                    blockY.U(:,a) = -1.0 * u_a;                    
                else
                    blockX.W(:,a) = w_a;
                    blockX.P(:,a) = p_a;
                    blockX.T(:,a) = t_a;
                    blockX.R(:,a) = r_a;
                    blockY.C(:,a) = c_a;
                    blockY.U(:,a) = u_a;
                end
                
                
                % We should do our randomization test internal to the PLS
                % algorithm, before we deflate.
                if self.opt.randomize_test.quick_return
                    self = struct;
                    self.t_a = t_a;
                    self.u_a = u_a;
                    self.itern = itern;
                    return;
                end
                if self.opt.randomize_test.use_internal
                    self = PLS_randomization_model(self, blockX, blockY);                    
                end
                
                self.stats.timing(a) = toc;
               

                % Now deflate the X-matrix and Y-matrix  
                if strcmp(self.model_type, 'mbpls') && not(self.opt.mbpls.deflate_X)
                else
                    X = X - t_a * p_a';
                end
                Y = Y - t_a * c_a';    
                
                % These are the Residual Sums of Squares (RSS); i.e X-X_hat
                row_SSX = ssq(X, 2);
                col_SSX = ssq(X, 1);
                row_SSY = ssq(Y, 2);
                col_SSY = ssq(Y, 1);

                blockX.stats.SPE(:,a) = sqrt(row_SSX/K);
                blockX.stats.R2k_cum(:,a) = 1 - col_SSX./blockX.stats.start_SS_col;
                blockX.stats.deflated_SS_col(:,a) = col_SSX(:);
                
                blockY.stats.R2k_cum(:,a) = 1 - col_SSY./blockY.stats.start_SS_col;
                

                % Cumulative R2 value for the whole block
                blockX.stats.R2(a) = 1 - sum(row_SSX)/sum(blockX.stats.start_SS_col);
                blockY.stats.R2(a) = 1 - sum(row_SSY)/sum(blockY.stats.start_SS_col);
 
                % VIP value: still to come
                
                blockX.A = a;
                blockY.A = a;

            end %looping on ``a``

            % Calculate Wstar = R = W inv(P'W) 
            blockX.R = blockX.W * inv(blockX.P' * blockX.W);
            blockY.data_pred_pp = blockX.T * blockY.C';
            blockY.data_pred = blockY.data_pred_pp ./ repmat(blockY.PP.scaling, N, 1) + ...
                                                  repmat(blockY.PP.mean_center, N, 1);
            
            % Calculate PLS regression coefficients
            blockX.beta = blockX.R * blockY.C';
            
            % Write the deflated arrays back to the block
            blockX.data = X;
            blockY.data = Y;

        end % ``fit_PLS``
       
        function [blockX, blockY] = apply_PLS(self, blockX, blockY, A, varargin)
            % Applies a PLS latent variable model on new data in ``blockX``
            % using the existing PLS model in ``self``.
            quick = false;
            if nargin == 5
                if strcmp(varargin{1}, 'quick')
                    quick = true;
                end
            end      
            
            tolerance = self.opt.tolerance;
            X = blockX.data;
            [N, K] = size(blockX.data);
            A = min(self.A, A);
            
            if quick
                % I know the code is duplicated to that below, but that's to
                % avoid a whole lot of "if quick, .... end" mess.
                for a = 1 : min(self.A, A)

                    % Regress each row in X on the p_a vector, and store the
                    % regression coefficient in t_a
                    t_a = regress_func(X, self.blocks{1}.W(:,a), blockX.has_missing);

                    % Deflate the X-matrix
                    X = X - t_a * self.blocks{1}.P(:,a)';                    
                    row_SSX = ssq(X, 2);
                    blockX.stats.SPE(:,a) = sqrt(row_SSX/K);
                    blockX.T(:,a) = t_a;
                end % looping on ``a`` latent variables
                blockX.data = X;
                blockX.A = a;
                return
            end % ``quick`` method
            
            
            % TODO(KGD): have a check here to verify the size of new data
            % is compatible with ``self``.
            
            % TODO(KGD): there is an inefficiency here: during
            % cross-validation with a = 1, 2, 3, ... A we are recomputing
            % over all components everytime.  We should allow incremental
            % improvements, as long as we receive the deflated matrices
            % from previous components.
            for a = 1:A
                tic;
                if a == 1                
                    blockX.stats.start_SS_col = ssq(X, 1);
                    blockY.stats.start_SS_col = ssq(blockY.data, 1);
                    initial_ssq_X = blockX.stats.start_SS_col;
                else
                    initial_ssq_X = ssq(X, 1);
                end
                if all(initial_ssq_X < tolerance)
                    warning('lvm:apply_PLS', 'There is no variance left in the data. Returning')
                    return
                end   
                
                % Baseline for all R^2 calculations
                start_SS_col = blockX.stats.start_SS_col;
                
                t_a = regress_func(X, self.blocks{1}.W(:,a), blockX.has_missing);

                % Deflate the X-matrix
                X = X - t_a * self.blocks{1}.P(:, a)';
                
                % These are the Residual Sums of Squares (RSS); i.e X - X_hat
                row_SSX = ssq(X, 2);
                col_SSX = ssq(X, 1);
                
                % Summary statistics for the X-block
                blockX.stats.SPE(:,a) = sqrt(row_SSX/K);
                blockX.stats.deflated_SS_col(:,a) = col_SSX(:);
                blockX.stats.R2k_cum(:,a) = 1 - col_SSX./start_SS_col;
                % Cumulative R2 value for the whole X-block
                blockX.stats.R2(a) = 1 - sum(row_SSX)/sum(start_SS_col);
                blockX.T(:,a) = t_a;
                blockX.A = a;
                
                % Populate values for the Y-block that we know of
                blockY.A = a;
                self.stats.timing(a) = toc;
            end % looping on ``a`` latent variables
            
            blockX.data = X; % Write the deflated array back to the block
            blockY.data_pred_pp = blockX.T(:, 1:A) * self.blocks{2}.C(:, 1:A)';
            blockY.data_pred = blockY.data_pred_pp ./ repmat(self.blocks{2}.PP.scaling, N, 1) + ...
                                          repmat(self.blocks{2}.PP.mean_center, N, 1);
        end % ``apply_PLS``        
  
        function disp(self)
            
            
            try
                risk_rate = self.stats.risk.rate;
                strength = self.stats.risk.strength;                
            catch ME
                risk_rate = ones(1, self.A) .* NaN;
                strength = ones(1, self.A) .* NaN;
            end
                                    
            for a = 0:(self.A+1)
                if strcmp(self.model_type, 'pca')
                    blockX = self.blocks{1};  
                    ncols = 3 + 1;
                    line_length = 2+8+12+ncols;
                    if a == 0
                        disp(repmat('-', 1, line_length))
                        fprintf('|%2s|%8s|%12s|\n', ...
                                'A', ...
                                '  R2X  ', ...
                                'Time(s)/Iter')
                        disp(repmat('-', 1, line_length))
                    elseif a==(self.A+1)
                        disp(repmat('-', 1, line_length))
                        fprintf('Overall R2X(cuml) = %6.2f%%\n', blockX.stats.R2(end)*100)
                    else
                        line_format = '|%2i|%8.2f|%6.4f / %3d|\n';
                        fprintf(line_format, ...
                                a, ...
                                blockX.stats.R2_a(a)*100, ...
                                self.stats.timing(a), ...
                                self.stats.itern(a))
                    end
                    
                elseif strcmp(self.model_type, 'npls')
                    blockX = self.blocks{1};
                    blockY = self.blocks{2};
                    ncols = 6+1;
                    line_length = 2+8+8+12+7+9+ncols;
                    if a == 0
                        disp(repmat('-', 1, line_length))
                        fprintf('|%2s|%8s|%8s|%12s|%7s|%9s|\n', ...
                                'A', ...
                                'R2X  ', ...
                                '  R2Y  ', ...
                                'Time(s)/Iter', ...
                                'Risk(%)',...
                                'Strength' ...
                                )
                        disp(repmat('-', 1, line_length))
                    elseif a==(self.A+1)
                        disp(repmat('-', 1, line_length))
                        fprintf('Overall R2X(cuml) = %6.2f%%\n', blockX.stats.R2(end)*100)
                        fprintf('Overall R2Y(cuml) = %6.2f%%\n', blockY.stats.R2(end)*100)

                    else
                        line_format = '|%2i|%8.2f|%8.2f|%6.4f / %3d|%7.1f|%9.2f|\n';
                        fprintf(line_format, ...
                                a, ...
                                blockX.stats.R2_a(a)*100, ...
                                blockY.stats.R2_a(a)*100, ...
                                self.stats.timing(a), ...
                                self.stats.itern(a), ...
                                risk_rate(a), ...
                                strength(a)*100)
                    end                    
                end % if pca-else-pls
            end % loop on ``a``
        end % ``disp``
        
        function plot(self, varargin)
            % Function to plot a model
            if strcmp(self.model_type, 'pca')
                blockX = self.blocks{1};
                blockY = [];
            elseif strcmp(self.model_type, 'npls')
                blockX = self.blocks{1};
                blockY = self.blocks{2};
            end
            
            if nargin == 1
                plottypes = 'summary';
                other_args = {};
            elseif nargin == 2
                plottypes = varargin{1};
                other_args = {};
            else
                plottypes = varargin{1};
                other_args = varargin(2:end);
            end
            
                
            
            if strcmpi(plottypes, 'all')
                plottypes = {'summary', 'obs', 'pred'};
            end
            
            if ~isa(plottypes, 'cell')
                plottypes = cellstr(plottypes);
            end            
            
           
            show_labels = true;

            
            % Iterate over all plots requested by the user
            for i = 1:numel(plottypes)
                plottype = plottypes{i};
                if strcmpi(plottype, 'summary')
                    plot_summary(self.model_type, blockX, blockY, show_labels)
                elseif strcmpi(plottype, 'obs')
                    plot_obs(self.model_type, blockX, blockY, show_labels)      
                elseif strcmpi(plottype, 'pred')
                    plot_pred(self.model_type, blockY)
                    
                elseif strcmpi(plottype, 'scores')
                    plot_scores(blockX, show_labels, other_args{:})
                    
                elseif strcmpi(plottype, 'spe')
                    plot_spe(blockX, show_labels, other_args{:})
                elseif strcmpi(plottype, 'R2')
                    plot_R2(blockX, show_labels);    
                elseif strcmpi(plottype, 'loadings')
                    if strcmp(self.model_type, 'npls')
                        plot_loadings(self.model_type, blockX, blockY, other_args);
                    else
                        plot_loadings(self.model_type, blockX, [], other_args);
                    end
                end
            end
                
        end % ``plot``
        
        function out = X(self)
            % Convenience function: returns the X block from a single-block 
            % PLS or PCA model
            out = cell(0);
            if self.B == 1
                out = self.blocks{1};
            elseif self.B==2 && strcmp(self.model_type(2:4), 'pls')
                out = self.blocks{1};
            elseif strcmp(self.model_type(1:2), 'mb')
                out = self.blocks(1:end-1);
            end
        end
        
        function out = Y(self)
            % Convenience function: returns the Y block for PLS models
            out = [];
            if self.B > 1
                out = self.blocks{end};
            end
        end
            
    end % methods
end % end classdef

%-------- Helper functions (usually used in 2 or more places). May NOT use ``self``
f

function stat = calculate_randomization_statistic(model_type, A, blocks)
    function v = robust_scale(a)
        % Using the formula from Mosteller and Tukey, Data Analysis and Regression,
        % p 207-208, 1977.
        n = numel(a);
        location = median(a);
        spread_MAD = median(abs(a-location));
        ui = (a - location)/(6*spread_MAD);    
        % Valid u_i values used in the summation:
        vu = ui.^2 <= 1;
        num = (a(vu)-location).^2 .* (1-ui(vu).^2).^4;
        den = (1-ui(vu).^2) .* (1-5*ui(vu).^2);
        v = n * sum(num) / (sum(den))^2;
    end
    
    % PCA randomization statistic
    if strcmp(model_type, 'pca')
        score_vector = blocks{1}.T(:, A);
        stat = sqrt(robust_scale(score_vector));
        
    elseif strcmp(self.model_type, 'npls')
    end    
end

function stats = update_permutation_statistics(stats, permuted_stats, num_G, nperm)
    % Updates the permutation statistics mean and variance, without having to
    % keep track of the original purmutation values.
    
    % How to combine variances without having the raw values?
    % A = 1:5;         B = 8:15;        C = [A B];
    % stdA = std(A);   stdB = std(B);   stdC = std(C);
    % meanA = mean(A); meanB = mean(B);
    % na = numel(A);   nb = numel(B);
    % A_ssq = stdA^2 * (na-1) + na*meanA^2;
    % B_ssq = stdB^2 * (nb-1) + nb*meanB^2;
    % C_mean = (meanA*na + meanB*nb)/(na+nb);
    % var_C = ((A_ssq + B_ssq) - (na+nb)*C_mean^2)/(na+nb-1)
    % which will match the actual: var(C)

    prev_ssq = stats.std_G^2 * (stats.nperm-1) + stats.nperm*stats.mean_G^2;
    curr_ssq = sum(permuted_stats.^2);
    stats.mean_G = (stats.mean_G*stats.nperm + mean(permuted_stats)*nperm)/(stats.nperm+nperm);        

    % Guard against getting a negative under the square root
    % Note: This approach quickly becomes inaccurate when using many rounds.
    stats.std_G = sqrt(((prev_ssq + curr_ssq) - (stats.nperm+nperm)*stats.mean_G^2)/(stats.nperm+nperm-1));
    stats.num_G_exceeded = stats.num_G_exceeded + num_G;
    stats.nperm = stats.nperm + nperm;    
end

function self = PCA_randomization_model(self, blockX)
    % See the technical details and reference under the
    % ``PCA_randomization_assess`` function.
    %
    % We will always use a minimum of permutations (see the lvm_opt.m) in the
    % randomization, perhaps more if we are on a fast computer and can
    % parallelize it.
    
    [N, K] = size(blockX.data);
    A = self.A;
    nperm = self.opt.randomize_test.permutations;
    self.opt.randomize_test.test_statistic = zeroexp([1, A], self.opt.randomize_test.test_statistic);
    self.opt.randomize_test.test_statistic(A) = calculate_randomization_statistic(self.model_type, self.A, {blockX});
        
    % TODO(KGD): ensure there is variation left in the X block to
    % support extracting components.  Right now that error check is left out
    % of the loop, in order to fit the randomized components quickly.
    
    X_original = self.blocks{1}.data;
    % Randomly permute all columns the first time:
    for k = 1:K
        rand('twister', k);
        self.blocks{1}.data(:,k) = self.blocks{1}.data(randperm(N),k);
    end
    
    capture_more_permutations = true;
    rounds = 0;
    stats = struct;
    stats.mean_G = 0;
    stats.std_G = 0;
    stats.num_G_exceeded = 0;
    stats.nperm = 0;
    while capture_more_permutations && (rounds < self.opt.randomize_test.max_rounds)
        % Will run the purmutation tests at least once, in a group of G, but
        % maybe more times, especially when the risk is borderline
        
        permuted_stats = zeros(nperm, 1);

        % Use the ``fit_PCA`` function internally, with a special flag to return early
        self.opt.randomize_test.quick_return = true;
        for g = 1:nperm
            % Set the random seed: to ensure we can reproduce calculations.
            % Shuffle the rows in Y randomly.
            rand('twister', g+rounds*nperm);
            self.blocks{1}.data(:, k) = self.blocks{1}.data(randperm(N), k); 
            k = k +1;
            if k > K
                k = 1;
            end

            % Calculate the "a"th component using this permuted X-matrix
            out = fit_PCA(self, self.blocks{1}, self.A);            
           
            % Next, calculate the statistic under test and store it            
            %stat = (out.u_a' * out.t_a) ./ (sqrt(out.t_a'*out.t_a) .* sqrt(out.u_a'*out.u_a));
            permuted_stats(g) = calculate_randomization_statistic(self.model_type, A, out);
        end
        num_G = sum(permuted_stats >= self.opt.randomize_test.test_statistic(A));
        rounds = rounds + 1;
        
        stats = update_permutation_statistics(stats, permuted_stats, num_G, nperm);
        
        % Assume we've got enough randomized values to make an accurate risk
        % assessment.
        capture_more_permutations = false;
        risk = stats.num_G_exceeded / stats.nperm * 100;
        bounds = self.opt.randomize_test.risk_uncertainty;
        if any( (risk > bounds(1)) && risk < bounds(2) )
            % Do another round of permutations to clarify risk level.
            capture_more_permutations = true;
        end

    end
    self.opt.randomize_test.risk_statistics{A} = stats;
    
    % Set the X-block back to its usual order
    self.blocks{1}.data = X_original;
    self.opt.randomize_test.quick_return = false;
end

function [self, terminate_adding] = PCA_randomization_assess(self)
    % Determines whether a new component should be added using a
    % randomization test.
    A = self.A;
    max_A = min(self.X.N, self.X.K);
    if not(isfield(self.opt.randomize_test, 'points'))
        self.opt.randomize_test.points = 0;
    end
    % Return if we've reached the maximum number of components supported by this model.
    if A >= max_A
        self.opt.randomize_test.last_worthwhile_A = A;
        terminate_adding = true;
        return
    end
     
    % We've already evaluated the risk-free cases, now let's evaulate the risky ones.
    %
    % 1.	Let risk = \frac{\text{number of}\,\,S_g\,\,\text{values exceeding}\,\,S_0}{G}
    % 
    %     *	If risk :math:`\geq 0.10`, then ``points = points + 2``, as there is a high risk, one in 10 chance, we are accepting a component that should not be accepted.
    % 
    %     *	or, if :math:`0.05 < \text{risk} < 0.10` then ``points = points + 1``  (moderately risky to accept this component) 
    % 
    %     *	finally, if :math:`risk \leq 0.05` then we accept the component without accumulating any points.
    % 
    % We stop adding components when the total risk points *accumulated on the current and all previous components* equals or exceeds 2.0.  
    % We revert to the component where we had a risk points of 1.0 or less and stop adding components.
    
    stats = self.opt.randomize_test.risk_statistics{A};
    current_risk = stats.num_G_exceeded  / stats.nperm;
    % Assess risk based on the number of violations in randomization
    if current_risk >= 0.10
        self.opt.randomize_test.points = self.opt.randomize_test.points + 2;
    elseif current_risk >= 0.05
        self.opt.randomize_test.points = self.opt.randomize_test.points + 1;
    elseif current_risk < 0.05
        self.opt.randomize_test.points = self.opt.randomize_test.points + 0;
    end
    if self.opt.randomize_test.points >= 2.0
        terminate_adding = true;
    else
        terminate_adding = false;
        if self.opt.randomize_test.points <= 1.0
            self.opt.randomize_test.last_worthwhile_A = A;
        end
    end
end

function [self, terminate_cv] = PCA_cross_validation(self, cv_pred)
% Determines whether the cross-validation rounds should terminate for PCA models.
    
    terminate_cv = false;
    A = self.A;
    max_A = min(self.X.N, self.X.K);
    if A >= max_A
        terminate_cv = true;
        return;
    end
    
    % Not sure why MATLAB is crashing here
    try
        G = self.opt.cross_val.groups;
    catch
        G = 5;
        % TODO(KGD): check why sometimes the groups subfield is not 
        % carried over.
        % G = getfield(self.opt.cross_val, 'groups');
    end
    N = 0;
    K = size(cv_pred{1}.blocks{1}.data, 2);
    PRESS_0 = 0;
    PRESS = zeros(1, A);
    for g = 1:G
        N = N + size(cv_pred{g}.blocks{1}.data, 1);
        PRESS = PRESS + sum(cv_pred{g}.blocks{1}.stats.deflated_SS_col,1);                    
        PRESS_0 = PRESS_0 + sum(cv_pred{g}.blocks{1}.stats.start_SS_col);
    end

    % Formula for nested models.  a=2 is actually a=1
    % below: to accomodate for the empty model.
    PRESS = [PRESS_0, PRESS];
    DOF = ((N-(0:A)).*(K-(0:A)));

    num_DOF = (DOF(A) - DOF(A+1));
    den_DOF = (N*K-DOF(A+1));
    num =(PRESS(A) - PRESS(A+1))/num_DOF;
    den = PRESS(A+1)/den_DOF;
    ratio = num/den;
    f_critical_99 = finv(0.99, num_DOF, den_DOF);
    if ratio > f_critical_99
        last_good_PC = A-1;
%         if self.opt.cross_val.strikes == 0
%             last_good_PC = A-1;
%         end
        self.opt.cross_val.strikes = self.opt.cross_val.strikes + 1;                        
    end
    %disp(['Ratio to F = ', num2str(ratio/f_critical_99)])


    % The correction factor: Wold, 1978, Technometrics, p 403.
    PRESS = PRESS ./ DOF;
    Q2 = 1 - PRESS(2:end) ./ (PRESS_0/(N*K));

    % If Q2(1) > 0.2, or if 
    if A == 1 && Q2 > 0.2
        self.opt.cross_val.strikes = self.opt.cross_val.strikes - 1;
    end
    if A > 1 && (Q2(end)-Q2(end-1)) >= 0.2
        if self.opt.cross_val.strikes > 0
            last_good_PC = A - 1;
            self.opt.cross_val.strikes = self.opt.cross_val.strikes - 1;
        end
    elseif A > 1 && (Q2(end)-Q2(end-1)) < 0.1
        last_good_PC = A - 1;
        self.opt.cross_val.strikes = self.opt.cross_val.strikes + 1;
    end
    if A == 1 && Q2 < 0.15
        self.opt.cross_val.strikes = self.opt.cross_val.strikes + 1;
        last_good_PC = A - 1;
    end


    if self.opt.cross_val.strikes > 1
        self.A = last_good_PC;
        terminate_cv = true;
    end                    
    if A == K
        terminate_cv = true;
    end

    show_plots = false;
    if show_plots
        subplot(1,3,1)
        cla();
        boxplot(R2)
        hold('on');
        plot(1:A, self.blocks{1}.stats.R2_a(1:A), 'r*', 'markersize', 10)
        title('R2')
        xlabel('Component')
        drawnow;

        subplot(1,3,2)
        bar(0:A, PRESS)
        title('PRESS')
        xlabel('Component')

        subplot(1,3,3)
        bar([self.blocks{1}.stats.R2(:) Q2(:)])
        title('Cumulative R2 and Q2')
        xlabel('Component')
    end
end

function self = PLS_randomization_model(self, blockX, blockY)
    % See the technical details and reference under the
    % ``PLS_randomization_assess`` function.
    %
    % We will always use a minimum of permutations (see the lvm_opt.m) in the
    % randomization, perhaps more if we are on a fast computer and can
    % parallelize it.
    
    [N, M] = size(blockY.data);
    A = self.A;
    nperm = self.opt.randomize_test.permutations;
    self.opt.randomize_test.test_statistic = zeroexp([1, A], self.opt.randomize_test.test_statistic);
    stat = (blockY.U(:,A)' * blockX.T(:,A)) ./ (sqrt(blockX.T(:,A)'*blockX.T(:,A)) .* sqrt(blockY.U(:,A)' * blockY.U(:,A)));
    %stat = abs((blockY.data' * blockX.T(:,A)) ./ (sqrt(blockX.T(:,A)'*blockX.T(:,A)) .* sqrt(blockY.data' * blockY.data)));
    self.opt.randomize_test.test_statistic(A) = stat; %abs((blockY.data' * blockX.T(:,A))/N);
    bounds = self.opt.randomize_test.risk_uncertainty;

    
    % TODO(KGD): ensure there is variation left in the X and Y blocks to
    % support extracting components.  Right now that error check is left out
    % of the loop, in order to fit the randomized components quickly.
    
    Y_original = self.blocks{2}.data;
    capture_more_permutations = true;
    rounds = 0;
    stats = struct;
    stats.mean_G = 0;
    stats.std_G = 0;
    stats.num_G_exceeded = 0;
    stats.nperm = 0;
    stats.stop_early = false;
    show_progress = self.opt.randomize_test.show_progress;
    if show_progress
        h = awaitbar(0, sprintf('Calculating risk of component %d', A));
    end
    while capture_more_permutations && (rounds < self.opt.randomize_test.max_rounds)
        % Will run the purmutation tests at least once, in a group of G, but
        % maybe more times, especially when the risk is borderline:
        
        permuted_stats = zeros(nperm, 1);

        % Use the ``fit_PLS`` function internally, with a special flag to
        % return early
        self.opt.randomize_test.quick_return = true;
        previous_max_iter = self.opt.max_iter;
        previous_tolerance = self.opt.tolerance;
        randomize_max_iter = 200;
        self.opt.max_iter = randomize_max_iter;
        self.opt.tolerance = 1e-2;
        itern = zeros(nperm,1);
        
        for g = 1:nperm
            perc = floor(g/nperm*100);
            num_G = stats.num_G_exceeded + sum(permuted_stats > self.opt.randomize_test.test_statistic(A));
            den_G = stats.nperm + g;
            if show_progress
                stats.stop_early = awaitbar(perc/100,h,sprintf('Risk of component %d. Risk so far=%d out of %d models. [%d%%]',A, num_G, den_G, perc));
                if stats.stop_early
                    nperm = g-1;
                    permuted_stats = permuted_stats(1:g-1);
                    rounds = Inf;  % forces it to exit
                    break; 
                end	
            end
            % Set the random seed: to ensure we can reproduce calculations.
            % Shuffle the rows in Y randomly.
            rand('twister', g+rounds*nperm);
            self.blocks{2}.data = self.blocks{2}.data(randperm(N), :); 

            % Calculate the "a"th component using this permuted Y-matrix, but
            % the unpermuted X-matrix.  
            out = fit_PLS(self, self.blocks{1}, self.blocks{2}, self.A);    
            
            if out.itern < randomize_max_iter
                % Next, calculate the statistic under test and store it            
                stat = (out.u_a' * out.t_a) ./ (sqrt(out.t_a'*out.t_a) .* sqrt(out.u_a'*out.u_a));
                permuted_stats(g) = stat; %(t_a' * self.blocks{2}.data)/N;
                itern(g) = out.itern;     
            end
        end
        self.opt.max_iter = previous_max_iter;
        self.opt.tolerance = previous_tolerance;
        
        num_G = sum(permuted_stats > self.opt.randomize_test.test_statistic(A));
        
               
        % TODO(KGD): put this into the above function
        prev_ssq = stats.std_G^2 * (stats.nperm-1) + stats.nperm*stats.mean_G^2;
        curr_ssq = nansum(permuted_stats.^2);
        stats.mean_G = (stats.mean_G*stats.nperm + nanmean(permuted_stats)*nperm)/(stats.nperm+nperm);        
        
        % Guard against getting a negative under the square root
        % This approach quickly becomes inaccurate when using many rounds.
        stats.std_G = sqrt(((prev_ssq + curr_ssq) - (stats.nperm+nperm)*stats.mean_G^2)/(stats.nperm+nperm-1));
        stats.num_G_exceeded = stats.num_G_exceeded + num_G;
        stats.nperm = stats.nperm + nperm;
        rounds = rounds + 1;
        
        
        % Assume we've got enough randomized values to make an accurate risk
        % assessment.
        capture_more_permutations = false;
        risk = stats.num_G_exceeded / stats.nperm * 100;
        if any( (risk > bounds(1)) && risk < bounds(2) )
            % Do another round of permutations to clarify risk level.
            capture_more_permutations = true;
        end        
    end
    if show_progress
        if ishghandle(h)
            close(h);
        end
    end
    self.opt.randomize_test.risk_statistics{A} = stats;
    
    % Set the Y-block back to its usual order
    self.blocks{2}.data= Y_original;
    self.opt.randomize_test.quick_return = false;
end

function [self, terminate_adding] = PLS_randomization_assess(self)
    % Determines whether a new component should be added using a
    % randomization test.
    % 
    % Wiklund, et al., J Chemometrics, 21, p427-439, 2007, "A randomization
    % test for PLS component selection". http://dx.doi.org/10.1002/cem.1086
    %
    % Also see: http://www.springerlink.com/content/u2u43772t603k7w7/
    % 
    % Based on results in that paper we will keep fitting components until
    % the deemed risk is too high.  The risk is quantified by a the number of
    % risk points.  After 2 points we stop adding components and revert to the
    % component where we had 1.0 points, or fewer.
    %
    % Points are cumulative. If we have 1.0 points, and we add a new
    % component, then the points from previous components are still retained.

    A = self.A;
    max_A = min(self.X.N, self.X.K);
    
    if not(isfield(self.opt.randomize_test, 'points'))
        self.opt.randomize_test.points = 0;
    end
    
    % Return if we've reached the maximum number of components supported by
    % this model.
    % Return if the user has clicked the "Stop adding" button during
    % randomization risk assessment.
    if A >= max_A || self.opt.randomize_test.risk_statistics{A}.stop_early
        self.opt.randomize_test.last_worthwhile_A = A;
        terminate_adding = true;
        return
    end
    
    % We've already evaluated the risk-free cases, now let's evaulate the risky ones.
    %
    % 1.	Let risk = \frac{\text{number of}\,\,S_g\,\,\text{values exceeding}\,\,S_0}{G}
    % 
    %     *	If risk :math:`\geq 0.08`, then ``points = points + 2``, as there is a high risk, one in 12 chance, we are accepting a component that should not be accepted.
    % 
    %     *	or, if :math:`0.03 < \text{risk} < 0.08` then ``points = points + 1``  (moderately risky to accept this component) 
    % 
    %     *	finally, if :math:`risk \leq 0.03` then we accept the component without accumulating any points, however, we might still add some points if the correlation, :math:`S_0` is small (see next step). 
    % 
    % 2.	Note that :math:`S_0` represents the correlation between :math:`t_a` and the :math:`u_a`, which is nothing more than a scaled version of the objective function of the PLS model, which each component is trying to maximize, subject to certain constraints.  We accumulate risk based on the strength of this correlation as follows:
    % 
    %     *	If :math:`S_0 \geq 0.50`, then we do not augment our risk, as this is a strong correlation
    % 
    %     *	Or, if :math:`0.35 < S_0 < 0.50`, then ``points = points + 0.5`` (weak correlation between :math:`t_a` and :math:`u_a`)
    % 
    %     *	Or, if :math:`S_0 \leq 0.35` then ``points = points + 1.0`` (very weak correlation between :math:`t_a` and :math:`u_a`)
    % 
    % We stop adding components when the total risk points *accumulated on the current and all previous components* equals or exceeds 2.0.  We revert to the component where we had a risk points of 1.0 or less and stop adding components.
    
    stats = self.opt.randomize_test.risk_statistics{A};
    current_risk = stats.num_G_exceeded  / stats.nperm;
    correlation = self.opt.randomize_test.test_statistic(A);
    current_points = self.opt.randomize_test.points;
    % Assess risk based on the number of violations in randomization
    if current_risk >= 0.10
        self.opt.randomize_test.points = self.opt.randomize_test.points + 2;
    elseif current_risk >= 0.05
        self.opt.randomize_test.points = self.opt.randomize_test.points + 1;
    elseif current_risk >= 0.01
        self.opt.randomize_test.points = self.opt.randomize_test.points + 0;
    elseif current_risk < 0.01
        self.opt.randomize_test.last_worthwhile_A = A;
        %self.opt.randomize_test.points = max(0, self.opt.randomize_test.points - 1.0);
        self.opt.randomize_test.risk_statistics{A}.points = self.opt.randomize_test.points - current_points;
        terminate_adding = false;
        return
    end
    
    % Assess risk also based on the strength of the t_a vs u_a correlation
    if correlation >= 0.5
        self.opt.randomize_test.points = self.opt.randomize_test.points + 0;
    elseif correlation >= 0.35
        self.opt.randomize_test.points = self.opt.randomize_test.points + 0.5;
    else
        self.opt.randomize_test.points = self.opt.randomize_test.points + 1.0;
    end
    
    self.opt.randomize_test.risk_statistics{A}.points = self.opt.randomize_test.points - current_points;
    
    if self.opt.randomize_test.points >= 2.0
        terminate_adding = true;
    else
        terminate_adding = false;
        if self.opt.randomize_test.points <= 1.0
            self.opt.randomize_test.last_worthwhile_A = A;
        end
    end
end

function [self, terminate_cv] = PLS_cross_validation(self, cv_pred)
    % Determines whether the cross-validation rounds should terminate for
    % PLS models.
    %
    % NOTE: I do not trust this code for cross-validation yet.  Not fully
    % validated.  KGD, 18 January 2011.
    
    terminate_cv = false;
    A = self.A;
    max_A = min(self.X.N, self.X.K);
    if A >= max_A
        terminate_cv = true;
        return;
    end
    
    G = self.opt.cross_val.groups;
    N = 0;
    PRESS = 0;
    for g = 1:G
        n_g = size(cv_pred{g}.blocks{2}.data, 1);
        N = N + n_g;
        predicted_Y = cv_pred{g}.blocks{2}.data_pred_pp;
        starting_Y_no_components = cv_pred{g}.blocks{2}.data;
        error = starting_Y_no_components - predicted_Y;
        PRESS_g = ssq(error);
        PRESS = PRESS + PRESS_g;     
    end
    if A == 1
        PRESS_0 = 0;
        for g = 1:G
            starting_Y_no_components = cv_pred{g}.blocks{2}.data;
            PRESS_0 = PRESS_0 + ssq(starting_Y_no_components);
        end
        self.opt.cross_val = setfield(self.opt.cross_val, 'PRESS_0', PRESS_0);
    end
    self.opt.cross_val = setfield(self.opt.cross_val, 'PRESS',  [self.opt.cross_val.PRESS PRESS]);
    
    K = self.X.K;
    DOF = ((N-(0:A)).*(K-(0:A)));
    num_DOF = (DOF(A) - DOF(A+1));
    den_DOF = (N*K-DOF(A+1));
    PRESS = [self.opt.cross_val.PRESS_0 self.opt.cross_val.PRESS];
    num =(PRESS(A) - PRESS(A+1))/num_DOF;
    den = PRESS(A+1)/den_DOF;
    ratio = num/den;
    f_critical_99 = finv(0.99, num_DOF, den_DOF);
    if ratio > f_critical_99
        if self.opt.cross_val.strikes == 0
            last_good_PC = A-1;
        end
        self.opt.cross_val = setfield(self.opt.cross_val, 'strikes',  self.opt.cross_val.strikes + 1);

    end
    disp(['Ratio to F = ', num2str(ratio/f_critical_99)])

    Q2 = 1 - self.opt.cross_val.PRESS ./ self.opt.cross_val.PRESS_0;
   	R2 = self.blocks{2}.stats.R2;
    disp('Q2');disp(Q2(:)')
    disp('R2');disp(R2(:)')
end

%-------- The major plot types: summary, observation, variable and prediction plots
function plot_summary(model_type, blockX, blockY, show_labels)
    figure('Color', 'white')
    nrow = 2;
    ncol = 2;    
    ax = 1;
    
    % t1-t2 scatter plot with ellipse
    if blockX.A >= 2      
        subplot(nrow, ncol, ax);
        plot_score_scatterplot(blockX, 1, 2, show_labels);
        ax = ax + 1;
    end
    
    subplot(nrow, ncol, ax);
    ax = ax + 1;
    if strcmp(model_type, 'pca')
        plot_loadings_scatterplot(blockX.P, [], 'P', 1, 2);
    else
        plot_loadings_scatterplot(blockX.R, blockY.C, 'W*C', 1, 2);
    end
   
    subplot(nrow, ncol, ax);
    plot_Hotellings_T2_lineplot(blockX, true, show_labels);
    ax = ax + 1;
    
    % SPE plot
    subplot(nrow, ncol, ax);
    plot_SPE_lineplot(blockX, true, show_labels);


    % Obs-vs-predicted for most variable Y or R2X_k bar plot
    % for PCA
end

function plot_obs(model_type, blockX, blockY, show_labels)
% Currently plots the following "observation"-wise plots
% 1. t_1 score plot
% 2. t_2 score plot
% 3. T2 plot
% 4. SPE plot

    figure('Color', 'white')
    nrow = 2;
    ncol = 2;    
    ax = 1;
    
    % t1 line plot
    if blockX.A >= 1        
        subplot(nrow, ncol, ax);
        plot_score_lineplot(blockX, 1, show_labels);
        ax = ax + 1;
    end
    
    % t2 line plot
    if blockX.A >= 2
        subplot(nrow, ncol, ax);
        plot_score_lineplot(blockX, 2, show_labels);
        ax = ax + 1;
    end

    % Hotelling's T2 plot
    subplot(nrow, ncol, ax);
    plot_Hotellings_T2_lineplot(blockX, true, show_labels);
    ax = ax + 1;
    
    % SPE plot
    subplot(nrow, ncol, ax);
    plot_SPE_lineplot(blockX, true, true);
    
    if isa(blockY, 'block') && strcmpi(model_type, 'npls')
        figure('Color', 'white')
        nrow = 2;
        ncol = 2;    
        for ax = 1:min(blockX.A, 4)
            subplot(nrow, ncol, ax)
            plot_score_parity_plot(blockX.T, blockY.U, 't', 'u', ax, ax);
        end
    end
end

function plot_spe(blockX, show_labels, varargin)

    % Defaults: show all scores
    PCs = 1:blockX.A;
    % Defaults: don't show scores over time
    %if strcmp(blockX.block_type, 'batch')
        batches = [];
    %end
    
    for i = 1:numel(varargin)
        if strcmpi(varargin{i}(1), 'a')
            PCs = varargin{i}(2);
        end
        if strcmpi(varargin{i}(1), 'batch')
            batches = varargin{i}(2);
        end
    end
    
    if not(isempty(batches))
        plot_batch_trajectories(blockX, 'spe', batches);
        return
    end

    if blockX.A == 0
        return    
    end
    plot_SPE_lineplot(blockX, true, show_labels);    
    
end

function plot_scores(blockX, show_labels, varargin)

    % Defaults: show all scores
    PCs = 1:blockX.A;
    % Defaults: don't show scores over time
    %if strcmp(blockX.block_type, 'batch')
        batches = [];
    %end
    
    for i = 1:numel(varargin)
        if strcmpi(varargin{i}(1), 'a')
            PCs = varargin{i}(2);
        end
        if strcmpi(varargin{i}(1), 'batch')
            batches = varargin{i}(2);
        end
    end
    
    if not(isempty(batches))
        plot_batch_trajectories(blockX, 'scores', batches);
        return
    end

    if blockX.A == 0
        return
    else
        figure('Color', 'White');
        ax = 1;
    end
    
    if blockX.A == 1        
        plot_score_lineplot(blockX, 1, show_labels);
   
    elseif blockX.A == 2
        nrow = 1;
        ncol = 2;
        subplot(nrow, ncol, ax);
        plot_score_scatterplot(blockX, 1, 2, show_labels);
        ax = ax + 1;
        
        % t1-t2 scatter plot with ellipse
        subplot(nrow, ncol, ax);
        plot_Hotellings_T2_lineplot(blockX, true, show_labels);
    elseif blockX.A >= 3
        nrow = 2;
        ncol = 2;
        
        subplot(nrow, ncol, ax);
        plot_score_scatterplot(blockX, 1, 2, show_labels);
        ax = ax + 1;
        subplot(nrow, ncol, ax);
        plot_score_scatterplot(blockX, 3, 2, show_labels);
        ax = ax + 1;
        subplot(nrow, ncol, ax);
        plot_score_scatterplot(blockX, 1, 3, show_labels);
        ax = ax + 1;
        subplot(nrow, ncol, ax);
        plot_Hotellings_T2_lineplot(blockX, true, show_labels);
    end    
end

function plot_loadings(model_type, blockX, blockY, varargin)
    if nargin > 3
        which_loadings = varargin{1};
    end
    if nargin <= 3 || isempty(which_loadings)
        if blockX.A >= 2
            which_loadings = [1, 2];
        elseif blockX.A == 1
            which_loadings = 1;
        else
            return;
        end
        
    end

    if strcmpi(blockX.block_type, 'ordinary')
        figure('Color', 'white');
        if strcmp(model_type, 'pca')
            nrow = 1;
        else
            nrow = 2;
        end
        ncol = 2; 
        ax = 1;
        subplot(nrow, ncol, ax);
        ax = ax + 1;
        if strcmp(model_type, 'pca')
            plot_loadings_scatterplot(blockX.P, [], 'P', which_loadings(1), which_loadings(2));
        else
            plot_loadings_scatterplot(blockX.R, blockY.C, 'W*C', which_loadings(1), which_loadings(2));
        end

        subplot(nrow, ncol, ax);
        ax = ax + 1;
        plot_R2X_per_variable_barplot(blockX, 'X');
        if strcmp(model_type, 'pca')
            return
        end

        subplot(nrow, ncol, ax);
        ax = ax + 1;
        plot_R2X_per_variable_barplot(blockY, 'Y');

        M = blockY.K;
        if M<=3
            nrow = 1;
            ncol = M;
        else
            nrow = 2;
            ncol = 2;
        end

        for m = 1:M  
            if ~mod(m-1, nrow+ncol)
                figure('Color', 'white');
                ax = 1;
            end
            subplot(nrow, ncol, ax);        
            title_str = ['Coefficients for variable ', num2str(m)];
            plot_variable_barplot(blockX.beta(:, m), title_str);
            ax = ax + 1;            
        end

    end
end

function plot_R2(blockX, show_labels)
    if blockX.A == 0 || strcmp(blockX.block_type, 'ordinary')
        return
    end
    
    % Only for batch models
    nSamples = blockX.J;            % Number of samples per tag
    %nTags = blockX.nTags;           % Number of tags in the batch data
    %tagNames = char(blockX.tagnames);
    
    hF = figure('Color', 'white');
    hA = subplot(1, 2, 1);
    hB = subplot(1, 2, 2);
    R2_tags_all_PC = zeros(blockX.nTags, blockX.A);
    
    for a = 1:blockX.A
        
        % These numbers don't agree with Nomikos's thesis
        data_num = blockX.stats.deflated_SS_col(:,a);
        data_den = blockX.stats.start_SS_col(:);
        
        temp = reshape(data_num, blockX.nTags, nSamples);
        data_num_time_grouped_per_variable = nansum(temp, 2);
        data_num_variable_grouped_by_time = nansum(temp, 1);
        temp = reshape(data_den, blockX.nTags, nSamples);
        data_den_time_grouped_per_variable = nansum(temp, 2);
        data_den_variable_grouped_by_time = nansum(temp, 1);
        
        R2_time_grouped_per_variable = 1- data_num_time_grouped_per_variable./data_den_time_grouped_per_variable;
        R2_variable_grouped_by_time = 1- data_num_variable_grouped_by_time./data_den_variable_grouped_by_time;
        
        R2_tags_all_PC(:,a) = R2_time_grouped_per_variable;
        % TODO(KGD): investigate why the above line (correct), does not match
        % with the line below: 
        %                          data = blockX.stats.R2k_a(:, a);
        
        set(hF, 'CurrentAxes', hB)
        plot(R2_variable_grouped_by_time)
        hold on
    end
    axis tight
    grid on
    title('R2 at each time step (X-space)')
    
    set(hF, 'CurrentAxes', hA)
    R2_tags_all_PC = R2_tags_all_PC';
    if blockX.A > 1
        R2_tags_all_PC = [R2_tags_all_PC(1,:); diff(R2_tags_all_PC)];
    end
    hBars = barh(R2_tags_all_PC', 'stacked');
    extent = axis;
    axis tight
    for a = 1:blockX.A
        set(hBars(a),'FaceColor', 'none')
    end
    
    for k = 1:blockX.nTags
        text(0.05, k, blockX.tagnames{k}, 'Rotation', 0, 'FontWeight', 'bold', 'FontSize', 12)
    end
    %axis([0, blockX.nTags+1, extent(3), extent(4)])
    axis([extent(1), extent(2), 0, blockX.nTags+1])
    title('R2 for each X-space variable')
    xlabel(sprintf('R2 per tag, per component [%d components]', blockX.A))
    set(hA, 'YDir', 'reverse')
end

function plot_pred(model_type, blockY)
% Plots all observed-vs-predicted plots; one for each y-variable.
    if ~strcmp(model_type, 'npls')
        return
    end
    M = blockY.K;
    if M<=3
        nrow = 1;
        ncol = M;
    else
        nrow = 2;
        ncol = 2;
    end

    for m = 1:M  
        if ~mod(m-1, nrow+ncol)
            figure('Color', 'white');
            ax = 1;
        end
        subplot(nrow, ncol, ax);
        observed = blockY.data_raw(:,m);
        predicted = blockY.data_pred(:, m);
        label = ['Variable ', num2str(m)];
        plot_obs_pred_plot(observed, predicted, label);
        ax = ax + 1;            
    end

end

%-------- Sub-functions: called by `plot_summary`, `plot_obs`, `plot_var`, `plot_pred`
function h_t = plot_score_lineplot(block, score, show_labels)
% Plots a line plot of t_score, where ``score`` is an integer: 1, 2, ... A.
    h_t = plot(block.T(:,score), '.-'); hold on
    plot([0 block.N], [0 0], 'k-', 'linewidth', 2)
    plot([0 block.N], [1 1]*block.lim.t(score), 'r--', 'linewidth', 1)
    plot([0 block.N], [1 1]*-block.lim.t(score), 'r--', 'linewidth', 1)
    title(['t_', num2str(score), ' score plot'])
    if show_labels
        for n = 1:block.N
            text(n+0.5, block.T(n,score), num2str(n));
        end
    end
    axis tight
end

function h_t = plot_Hotellings_T2_lineplot(block, normalized, show_labels)    
% Plots Hotelling's T2 lineplot with the confidence limit (95%).
% Uses all A components.
    
    if normalized
        scale = block.lim.T2(block.A);
        words = 'Normalized Hotelling''s ';
    else
        scale = 1.0;
        words = 'Hotelling''s ';
    end
        
    h_t = plot(block.stats.T2(:, block.A)./scale, '.-'); hold on
    plot([0 block.N], [0 0], 'k-', 'linewidth', 2)
    plot([0 block.N], [1 1]*block.lim.T2(block.A)./scale, 'r--', 'linewidth', 1)
    title([words, 'T^2 using ', num2str(block.A), ' components'])
    if show_labels
        for n = 1:block.N
            text(n+0.5, block.stats.T2(n, block.A)./scale, num2str(n))
        end
    end
    axis tight
    grid('on')
end

function h_t = plot_SPE_lineplot(block, normalized, show_labels)  
% Plots SPE lineplot with the confidence limit (95%).
% Uses all A components.
    figure('Color', 'white')
    if normalized
        scale = block.lim.SPE(block.A);
        words = 'Normalized squared ';
    else
        scale = 1.0;
        words = 'Squared ';
    end
    
    h_t = plot(block.stats.SPE(:, block.A)./scale, '.-'); hold on
    plot([0 block.N], [0 0], 'k-', 'linewidth', 2)
    plot([0 block.N], [1 1]*block.lim.SPE(block.A)./scale, 'r--', 'linewidth', 1)
    title([words, 'prediction error (SPE) using ', num2str(block.A), ' components'])
    if show_labels
        for n = 1:block.N
            text(n+0.5, block.stats.SPE(n, block.A)./scale, num2str(n))
        end
    end
    axis tight
    grid('on')
end
    
function h_t = plot_score_scatterplot(block, ah, av, show_labels)    
% Plots the t_ah vs t_av score scatterplot with the confidence limit (95%).
% Uses component ``ah`` on the horizontal axis and ``av`` on the vertical
% axis.
    h_t = plot(block.T(:,ah), block.T(:,av), 'k.'); hold on
    [x95, y95] = ellipse_coordinates(block.S(ah), block.S(av), block.lim.T2(end), 100);
    plot(x95, y95, 'r--', 'linewidth', 1)
    title('Score target plot')
    grid on
    %axis equal 
    xlabel(['t_', num2str(ah)])
    ylabel(['t_', num2str(av)])    
    extent = axis;
    hd = plot([0, 0], [-1E10, 1E10], 'k', 'linewidth', 2);
    set(hd, 'tag', 'vline', 'HandleVisibility', 'off');
    hd = plot([-1E50, 1E50], [0, 0], 'k', 'linewidth', 2);
    set(hd, 'tag', 'hline', 'HandleVisibility', 'off');
    xlim(extent(1:2))
    ylim(extent(3:4)) 
    delta = 0.01*diff(extent(1:2));
    if show_labels
        for n = 1:block.N
            text(block.T(n,ah)+delta, block.T(n,av), num2str(n))
        end
    end
end

function [x, y] = ellipse_coordinates(s_h, s_v, T2_limit_alpha, n_points)
    
%     We want the (t_horiz, t_vert) cooridinate pairs that form the T2 ellipse.
% 
%     Inputs: s_h (std deviation of the score on the horizontal axis)
%             s_v (std deviation of the score on the vertical axis)
%             T2_limit_alpha: the T2_limit at the alpha confidence value
% 
%     Equation of ellipse in canonical form (http://en.wikipedia.org/wiki/Ellipse)
% 
%      (t_horiz/s_h)^2 + (t_vert/s_v)^2  =  T2_limit_alpha
% 
%      s_horiz = stddev(T_horiz)
%      s_vert  = stddev(T_vert)
%      T2_limit_alpha = T2 confidence limit at a given alpha value (e.g. 99%)
% 
%     Equation of ellipse, parametric form (http://en.wikipedia.org/wiki/Ellipse)
% 
%     t_horiz = sqrt(T2_limit_alpha)*s_h*cos(t) 
%     t_vert  = sqrt(T2_limit_alpha)*s_v*sin(t) 
% 
%     where t ranges between 0 and 2*pi
% 
%     Returns `n-points` equi-spaced points on the ellipse.

    h_const = sqrt(T2_limit_alpha) * s_h;
    v_const = sqrt(T2_limit_alpha) * s_v;
    dt = 2*pi/(n_points-1);
    steps = 0:(n_points-1);
    x = cos(steps*dt)*h_const;
    y = sin(steps*dt)*v_const;
end

function h_t = plot_score_parity_plot(scoresA, scoresB, label_h, label_v, ah, av)
% Scores partity plot, such as t_1 vs u_1.
    ms = 15;
    h_t = plot(scoresA(:,ah), scoresB(:, av), 'k.', 'Markersize', ms);     
    hold('on')
    title('Score parity plot')
    grid on
    axis equal 
    xlabel([label_h, '_', num2str(ah)])
    ylabel([label_v, '_', num2str(av)])
    extent = axis;
    hd = plot([-1E10, 1E10], [-1E10, 1E10], 'k', 'linewidth', 2);
    set(hd, 'tag', 'hline', 'HandleVisibility', 'off')    
    xlim(extent(1:2))
    ylim(extent(3:4))
end

function h_t = plot_loadings_scatterplot(loadingsA, loadingsB, label, ah, av)
% Superimposes loadings from ``loadingsA`` and ``loadingsB`` on the same plot, but
% with difference colours.  Uses ``labels`` for the axis labels and shows
% column ``ah`` on the horizontal axis; column ``av`` on the vertical axis.
    ms = 15;
    h_t = plot(loadingsA(:,ah), loadingsA(:,av), 'k.', 'Markersize', ms); 
    hold on
    if ~isempty(loadingsB)
        h_t = plot(loadingsB(:,ah), loadingsB(:,av), 'r.', 'Markersize', ms);
    end
    title('Loadings plot')
    grid on
    axis equal 
    xlabel([label, '_', num2str(ah)])
    ylabel([label, '_', num2str(av)])
    extent = axis;
    hh = plot([-1E100, 1E100], [0, 0], 'k', 'linewidth', 2);
    set(hh, 'tag', 'hline', 'HandleVisibility', 'off')
    hv = plot([0, 0], [-1E10, 1E10], 'k', 'linewidth', 2);
    set(hv, 'tag', 'hline', 'HandleVisibility', 'off')
    xlim(extent(1:2))
    ylim(extent(3:4))
end

function h_t = plot_obs_pred_plot(observed, predicted, label)
% Plots the observed-vs-predicted plot for a particular variable.
    ms = 15;
    h_t = plot(observed, predicted, 'k.', 'Markersize', ms);     
    hold('on')
    title('Observed vs predicted parity plot')
    grid on
    axis equal 
    xlabel(['Observed: ', label])
    ylabel(['Predicted: ', label])
    extent = axis;
    min_ex = min(extent([1,3]));
    max_ex = min(extent([2,4]));
    delta = (max_ex - min_ex);
    min_ex_l = min_ex - delta*1.5;
    max_ex_l = max_ex + delta*1.5;
    hd = plot([min_ex_l, max_ex_l], [min_ex_l, max_ex_l], 'k', 'linewidth', 2);
    set(hd, 'tag', 'hline', 'HandleVisibility', 'off')    
    
    RMSEP = sqrt(mean((observed - predicted).^2));
    text(min_ex + 0.05*delta, max_ex - 0.05*delta, sprintf('RMSEP = %0.4g', RMSEP))
    
    xlim([min_ex-0.1*delta, max_ex+0.1*delta])
    ylim([min_ex-0.1*delta, max_ex+0.1*delta])

end

function h_t = plot_R2X_per_variable_barplot(block, label)
    h_t = bar(block.stats.R2k_cum(:, block.A), 'k'); 
    hold('on')    
    title(['R^2 for each ', label ' variable'])    
    if block.K > 1
        plot([0 block.K+1], [0 0], 'k-', 'linewidth', 2)
        axis tight
    end
end

function h_t = plot_variable_barplot(data, title_label, varargin)
    % Plots a bar-plot for variable-wise data
    if nargin >= 3
        xlabel_str = varargin{1};
    else
        xlabel_str = '';
    end
    if nargin >= 4
        ylabel_str = varargin{2};
    else
        ylabel_str = '';
    end
        
    h_t = bar(data, 'k'); 
    hold('on')    
    title(title_label)  
    xlabel(xlabel_str)
    ylabel(ylabel_str)
    if numel(data) > 1
        plot([0 numel(data)+1], [0 0], 'k-', 'linewidth', 2)
        axis tight
    end
end

function h_t = plot_batch_trajectories(block, traj_type, batch_num)

    % Plots instantaneous trajectories of various types
    
    % plot_batch_traj('scores', 1, [37])   % 
    % plot_batch_traj('SPE', 3, [35]) 
    % plot_batch_traj('T2', 3, [32]) 
    
    figure('Color', 'white')
    h_t = axes();
    batch_num = batch_num{1};
    if strcmpi(traj_type, 'scores')
        data = squeeze(block.T_j(batch_num,:,:));
        limits = block.lim.t_j;
        for a = 1:block.A
            subplot(block.A, 1, a)
            plot(data(a,:), 'k', 'linewidth', 2)
            hold on
            plot(limits(:,a), 'r-.', 'linewidth', 2)
            plot(-limits(:,a), 'r-.', 'linewidth', 2)
            ylabel(['Score: ', num2str(a)])
            xlabel('Batch time')
            title(['Time-varying score: ', num2str(a), ' for batch ', num2str(batch_num)])
            grid('on')            
        end
    elseif strcmpi(traj_type, 'SPE')
        data = block.stats.SPE_j(batch_num,:);
        limits = block.lim.SPE_j;
                    
        plot(data(:) ./ limits, 'k')
        grid on
        hold on
        % Show normalized SPE
        plot([0 block.J+1], [1.0, 1.0], 'r-.', 'linewidth', 2)        
        title(['Time-varying SPE after ', num2str(block.A), ' components for batch ', num2str(batch_num)])
        ylabel('SPE')
        xlim([0, block.J+1])
        
        extent = axis;
        if extent(4) < 1.1
            axis([extent(1:3), 1.1])
        end
        
        
    elseif strcmpi(traj_type, 'T2')
        data = block.stats.T2_j(batch_num,:);
        limits = block.lim.T2;  % T2 limit is constant: just the usual T2 limit
                    
        plot(data(:), 'k', 'linewidth', 2)
        hold on
        plot(limits(:), 'r-.', 'linewidth', 2)
        title(['Time-varying T2 with all ', num2str(block.A), ' components'])
        ylabel('T2')
        xlabel('Batch time')
        grid('on')
    end
end

function SPE_limit = calculate_SPE_limit(SPE_values, alpha)
    % Given any number of SPE_value (whatever shape), it will calculate the
    % single SPE limit.  It unfolds the array and uses all entries to
    % calculate the limit, using the Nomikos and MacGregor approximation.
    % 
    % SPE_values are assumed to be calculated from Q = e.*e
    % i.e. no square rooting and dividing by a constant.
    var_SPE = var(SPE_values(:));
    avg_SPE = mean(SPE_values(:));
    chi2_mult = var_SPE/(2.0 * avg_SPE);
    chi2_DOF = (2.0*avg_SPE^2)/var_SPE;
    SPE_limit = chi2_mult * my_chi2inv(alpha, chi2_DOF);
end