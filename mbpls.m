% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Multi-block PLS models

% Derived from the general class of "multiblock latent variable models", mblvm
classdef mbpls < mblvm
    properties (SetAccess = protected)
        Y = [];
        Yhat = [];
        YPP = [];
    end
        
    methods 
        function self = mbpls(varargin)
            
            % Pull out the Y-block and place it into a specific variable
            blocks_given = varargin{1};
            for b = 1:numel(blocks_given)
                if strcmpi(blocks_given{b}.name, 'y')
                    Y_block = blocks_given{b};
                    blocks_given(b) = []; % delete the Y-block
                    
                    %Ny = shape(self.Y, 1);
                    %if self.N ~= Ny
                    %    warning('mbpls:merge_blocks', 'The number of observations in all blocks must match.')
                    %end
                end 
            end
            varargin{1} = blocks_given;
            self = self@mblvm(varargin{:});  
            self.Y = Y_block;
            self.Yhat = copy(Y_block);
            self.Yhat.data = self.Yhat.data .* 0;
            self.M = shape(self.Y, 2);
            
        end % ``mbpls``        
        
        % Superclass abstract method implementation
        function self = expand_storage(self, varargin)
            if numel(self.YPP) == 0
                self.YPP = struct('mean_center', [], ...
                    'scaling', [], ...
                    'is_preprocessed', false);
            end
        end % ``expand_storage``
        
        % Superclass abstract method implementation
        function self = preprocess_extra(self)
            if ~self.YPP.is_preprocessed
                [self.Y, PP_block] = self.Y.preprocess();
                self.YPP.is_preprocessed = true;
                self.YPP.mean_center = PP_block.mean_center;
                self.YPP.scaling = PP_block.scaling;                    
            end
            
        end
        
        % Superclass abstract method implementation
        function self = calc_model(self, A)
            % Fits a PLS latent variable model.
            %  
            % We assume the data are merged and preprocessed already.
            % Must also calculate all summary statistics for each block.
            
            % Perform ordinary missing data PLS on the merged block of data
            if not(isempty(self.Y.mmap)) > 1 && any(sum(self.Y.mmap, 2) == 0)
                warning('mbpls:calc_model', ...
                        ['Cannot handle the case yet where the entire '...
                         'observation in Y-matrix is missing.  Please '...
                         'remove those rows and refit model.'])
            end
            
            which_components = max(self.A+1, 1) : A;            
            for a = which_components
                
                start_time = cputime;
                % Baseline for all R2 calculations and variance check
                if a == 1             
                    ssq_before = ssq(self.data, 1);
                    self.split_result(ssq_before, 'stats', 'start_SS_col');
                    
                    ssq_Y_before = ssq(self.Y.data, 1);
                    self.super.stats.ssq_Y_before = ssq_Y_before;
                else
                    ssq_before = ssq(self.data, 1);
                    ssq_Y_before = ssq(self.Y.data, 1);
                end
                
                if all(ssq_before < self.opt.tolerance)
                    warning('mbpls:calc_model', 'There is no variance left in the X-data')
                end
                if all(ssq_Y_before < self.opt.tolerance)
                    warning('mbpls:calc_model', 'There is no variance left in the Y-data')
                end 
                                 
                % Converge onto a single component
                [t_a, p_a, c_a, u_a, w_a, itern] = mbpls.single_block_PLS(self.data, self.Y.data, self, a, self.has_missing); 
                
                % Flip the signs of the column vectors in P so that the largest
                % magnitude element is positive.
                % (Wold, Esbensen, Geladi, PCA, CILS, 1987, p 42
                %  http://dx.doi.org/10.1016/0169-7439(87)80084-9)
                [max_el, max_el_idx] = max(abs(w_a)); %#ok<ASGLU>
                if sign(w_a(max_el_idx)) < 1
                    p_a = -1.0 * p_a;
                    t_a = -1.0 * t_a;
                    c_a = -1.0 * c_a;
                    u_a = -1.0 * u_a;
                end                    
                
                self.model.stats.timing(a) = cputime - start_time;
                self.model.stats.itern(a) = itern;
                
                % Recover block information and store that.
                t_superblock = zeros(self.N, self.B);
                for b = 1:self.B
                    idx = self.b_iter(b);
                    X_portion  = self.data(:, idx);
                    
                    % Regress sub-columns of self.data onto the superscore
                    % to get the block weights.
                    w_b = regress_func(X_portion, u_a, self.has_missing);
                    
                    w_b = w_b / norm(w_b);
                    
                    % Block scores: regress rows of X onto the block loadings
                    t_b = regress_func(X_portion, w_b, self.has_missing);
                    t_superblock(:,b) = t_b;
                    %T_b_recovered{b}(:,a) = X_portion * w_b / (w_b'*w_b) / sqrt(K_b(b));
                    
                    
                    % Block loadings: that would have been used to deflate the
                    % X-blocks
                    p_b = regress_func(X_portion, t_a, self.has_missing);
                    
                    % Store the block scores, weights and loadings
                    self.T{b}(:,a) = t_b;
                    self.W{b}(:,a) = w_b;
                    self.P{b}(:,a) = p_b;
                   
                    % Store the SS prior to deflation 
                    X_portion_hat = t_a * p_b';
                    self.stats{b}.col_ssq_prior(:, a) = ssq(X_portion_hat,1);
                    
                    % VIP calculations
                    % -----------------                    
                    ssq_after = ssq(X_portion - X_portion_hat, 1);
                    VIP_temp = zeros(self.K(b), 1);
                    for a_iter = 1:a
                        denom = sum(self.stats{b}.start_SS_col - ssq_after);
                        self.stats{b}.VIP_f{a_iter,a} = sum(self.stats{b}.col_ssq_prior(:,a_iter)) /  denom;
                        % was dividing by /(sum(self.stats{b}.start_SS_col) - sum(ssq_after));
                        VIP_temp = VIP_temp + self.W{b}(:,a_iter) .^ 2 * self.stats{b}.VIP_f{a_iter,a} * self.K(b);
                    end
                    self.stats{b}.VIP_a(:,a) = sqrt(VIP_temp);
                
                    % Block T2 using the number of components calculated so far
                    [self.stats{b}.T2(:,a), S] = self.mahalanobis_distance(self.T{b}(:,1:a));
                    self.stats{b}.S = S;
                    
                end
                w_super = regress_func(t_superblock, u_a, false);
                w_super = w_super / norm(w_super);
                
                % Store the super-level results
                self.super.T_summary(:,:,a) = t_superblock;
                self.super.T(:,a) = t_a;
                self.super.W(:,a) = w_super;
                self.super.C(:,a) = c_a;
                self.super.U(:,a) = u_a;
                                
                % Now deflate the data matrix using the superscore
                self.data = self.data - t_a * p_a';
                
                % Make current predictions of Y using all available PCs
                Y_hat_update = t_a * c_a';
                self.Yhat.data = self.Yhat.data + Y_hat_update;
                self.Y.data = self.Y.data - Y_hat_update;
                ssq_Y_after = ssq(self.Y.data, 1)';
                self.super.stats.R2Yk_a(:,a) = 1 - ssq_Y_after ./ self.super.stats.ssq_Y_before';
                self.super.stats.R2Y(a) = 1 - sum(ssq_Y_after)/ sum(self.super.stats.ssq_Y_before);
                if a>1
                    self.super.stats.R2Y(a) = self.super.stats.R2Y(a) - sum(self.super.stats.R2Y(1:a-1), 2);
                    self.super.stats.R2Yk_a(:,a) = self.super.stats.R2Yk_a(:,a) - sum(self.super.stats.R2Yk_a(:,1:a-1), 2);
                end
                
                ssq_cumul = 0;
                ssq_before = 0;
                for b = 1:self.B
                    idx = self.b_iter(b);
                    
                    % X_portion has already been deflated by the current PC
                    X_portion = self.data(:, idx);
                    
                    % Calculate SPE
                    row_ssq = ssq(X_portion, 2);
                    self.stats{b}.SPE(:,a) = sqrt(row_ssq ./ numel(idx));
                    
                    % Calculate R2 per variable
                    col_ssq_remain = ssq(X_portion, 1)';
                    ssq_cumul = ssq_cumul + sum(col_ssq_remain);
                    ssq_before = ssq_before + sum(self.stats{b}.start_SS_col);
                    
                    self.stats{b}.R2Xk_a(:,a) = 1 - col_ssq_remain ./ self.stats{b}.start_SS_col';
                    self.stats{b}.R2b_a(1,a) = 1 - sum(col_ssq_remain) / sum(self.stats{b}.start_SS_col);
                    self.stats{b}.SSQ_exp(1,a) = sum(col_ssq_remain);
                    if a>1
                        self.stats{b}.R2Xk_a(:,a) = self.stats{b}.R2Xk_a(:,a) - sum(self.stats{b}.R2Xk_a(:,1:a-1), 2);
                        self.stats{b}.R2b_a(1,a) = self.stats{b}.R2b_a(1,a) - sum(self.stats{b}.R2b_a(1,1:a-1), 2);
                    end
                     
                    
                end
                
                
                % Cumulative R2 value for the whole component
                self.super.stats.R2X(a) = 1 - ssq_cumul/ssq_before;
                if a>1
                    self.super.stats.R2X(a) = self.super.stats.R2X(a) - sum(self.super.stats.R2X(1:a-1), 2);
                end
                
                % Store explained variance
                self.super.stats.SSQ_exp(1,a) = ssq_cumul;
                
                % Model summary SPE (not the superblock's SPE!), merely the
                % overall SPE from the merged model
                row_ssq_deflated = ssq(self.data, 2);
                self.super.SPE(:,a) = sqrt(row_ssq_deflated ./ sum(self.K));
                
                % Model summary T2 (not the superblock's T2!), merely the
                % overall T2 from the merged model
                [self.super.T2(:,a), S] = self.mahalanobis_distance(self.super.T(:,1:a));
                self.super.S = S;
                
                VIP_temp = zeros(sum(self.B), 1);
                for a_iter = 1:a
                    self.super.stats.VIP_f{a_iter,a} = self.super.stats.SSQ_exp(1,a_iter) / sum(self.super.stats.SSQ_exp);
                    VIP_temp = VIP_temp + self.super.W(:,a_iter) .^ 2 * self.super.stats.VIP_f{a_iter,a} * sum(self.B);
                end
                self.super.stats.VIP(1:self.B,a) = sqrt(VIP_temp);
                
                % Calculate the limits                
                self.calc_statistics_and_limits(a);
                
                self.A = a;
            end % looping on ``a`` latent variables
          
            
        end % ``calc_model``
    
        % Superclass abstract method implementation
        function limits_subclass(self)
            % Calculates the monitoring limits for a batch blocks in the model
            for b = 1:self.B                
            end
            
        end % ``calc_model_post``
        
        % Superclass abstract method implementation
        function state = apply_model(self, new, state, varargin) 
            % Applies a PCA model to the given ``block`` of (new) data.
            % 
            % TODO(KGD): allow user to specify ``A``
            
            which_components = 1 : min(self.A);
            for a = which_components
                
                initial_ssq_total = zeros(state.Nnew, 1);
                initial_ssq = cell(1, self.B);
                for b = 1:self.B                    
                    initial_ssq{b} = ssq(new{b}.data, 2);
                    initial_ssq_total = initial_ssq_total + initial_ssq{b};
                    if a==1                        
                        state.stats.initial_ssq{b} = initial_ssq{b};
                        state.stats.initial_ssq_total(:,1) = initial_ssq_total;
                    end
                end

                if all(initial_ssq_total < self.opt.tolerance)
                    warning('mbpca:apply_model', 'There is no variance left in one/some of the new data observations')
                end

                for b = 1:self.B
                    % Block score
                    state.T{b}(:,a) = regress_func(new{b}.data, self.W{b}(:,a), new{b}.has_missing);
                    state.T{b}(:,a) = state.T{b}(:,a) .* self.block_scaling(b);
                    % Transfer it to the superscore matrix
                    state.T_sb(:,b,a) = state.T{b}(:,a);
                end
                
                % Calculate the superscore, T_super
                state.T_super(:,a) = state.T_sb(:,:,a) * self.super.W(:,a);
                
                % Deflate each block: using the SUPERSCORE and the block loading
                for b = 1:self.B
                    deflate = state.T_super(:,a) * self.P{b}(:,a)';
                    state.stats.R2{b}(:,1) = ssq(deflate, 2) ./ state.stats.initial_ssq{b};
                    new{b}.data = new{b}.data - deflate;
                end
            end % looping on ``a`` latent variables
            state.Y_pred = state.T_super * self.super.C(:,1:a)';
            
            % Summary statistics for each block and the super level
            overall_variance = zeros(state.Nnew, 1);
            for b = 1:self.B
                block_variance = ssq(new{b}.data, 2);
                overall_variance = overall_variance + block_variance;
                state.stats.SPE{b} = sqrt(block_variance ./ self.K(b));
            end
            state.stats.super.R2(:,1) = 1 - overall_variance ./state.stats.initial_ssq_total;            
            state.stats.super.SPE(:,1) = sqrt(overall_variance ./ sum(self.K));
            state.stats.super.T2(:,1) = mblvm.mahalanobis_distance(state.T_super, self.super.S);
        end % ``apply_model``
        
        % Superclass abstract method implementation
        function summary(self)
            % Displays more information about ``self``
            
            % TODO(KGD): improve this by not displaying block info if B==1
            
            fprintf('R2 summary for %s (all as percentages)\n', self.model_type);
            w_char = zeros(self.B + 1,1);
            ncols = 2 + self.B + 1;  % A column, overall R2, and then all blocks + Y-block
            line_length = 3 + 8 + ncols;
            all_lines = '|%3i|%7.2f|';
            start_line = '|%3s|%7s|';
            block_names = cell(self.B + 1,1);
            for b = 1:self.B   % for the Y-block
                if strcmp(self.blocks{b}.name_type, 'auto')
                    % Drop off the "block-" part of the automatic block name
                    block_names{b} = self.blocks{b}.name(7:end);
                else
                    block_names{b} = self.blocks{b}.name;
                end
                
                w_char(b) = max(6, numel(block_names{b}));
                all_lines = [all_lines, '%', num2str(w_char(b)), '.2f|']; %#ok<AGROW>
                start_line = [start_line,  '%', num2str(w_char(b)), 's|']; %#ok<AGROW>                
                line_length = line_length + w_char(b);
            end
            if strcmp(self.Y.name_type, 'auto')
                % Drop off the "block-" part of the automatic block name
                block_names{b+1} = self.Y.name(7:end);
            else
                block_names{b+1} = self.Y.name;
            end
            
            w_char(b+1) = max(6, numel(block_names{b+1}));
            all_lines = [all_lines, '%', num2str(w_char(b+1)), '.2f|\n']; 
            start_line = [start_line,  '%', num2str(w_char(b+1)), 's|\n']; 
            line_length = line_length + w_char(b+1);
            
            disp(repmat('-', 1, line_length))
            fprintf_args = {' A ', ' All X ', block_names{:}};
            fprintf(start_line, fprintf_args{:});
            disp(repmat('-', 1, line_length))
            
            
            for a = 1:self.A
                fprintf_args = zeros(1, 2+self.B);
                fprintf_args(1:2) =  [a, self.super.stats.R2X(a)*100];
                for b = 1:self.B
                    fprintf_args(2+b) = self.stats{b}.R2b_a(a)*100;
                end
                fprintf_args(2+b+1) = self.super.stats.R2Y(a)*100;
                fprintf(all_lines, fprintf_args);
            end
            
            disp(repmat('-', 1, line_length))
            fprintf('Overall R2X(cumul) = %6.2f%%\n', sum(self.super.stats.R2X)*100)
            fprintf('Overall R2Y(cumul) = %6.2f%%\n', sum(self.super.stats.R2Y)*100)
            fprintf('Time to calculate (s): = [');
            for a= 1:self.A
                fprintf('%3.2f', self.model.stats.timing(a))
                if a ~= self.A
                    fprintf(', ')
                end
            end
            fprintf(']\n');
            
            fprintf('Number of iterations: = [');
            for a= 1:self.A
                fprintf('%3d', self.model.stats.itern(a))
                if a ~= self.A
                    fprintf(', ')
                end
            end
            fprintf(']\n');
            
        end
        
    end % end methods (ordinary)
    
    % These methods don't require a class instance
    methods(Static)
        function [t_a, p_a, c_a, u_a, w_a, itern] = single_block_PLS(X, Y, self, a, has_missing)
            % Extracts a PCA component on a single block of data, ``data``.
            % The model object, ``self``, should also be provided, for options.
            % The ``a`` entry is merely used to show which component is being
            % extracted in the progress bar.
            % The ``has_missing`` flag is used to indicate if any entries in 
            % ``dblock`` are missing.
            %
            %
            % [1] Höskuldsson, PLS regression methods, Journal of Chemometrics,
            %     2(3), 211-228, 1998, http://dx.doi.org/10.1002/cem.1180020306
            %
            % [2] Missing data: http://dx.doi.org/10.1016/B978-044452701-1.00125-3
            
            N = size(X, 1);
            if self.opt.show_progress
                h = awaitbar(0, sprintf('Calculating component %d', a));
            end
            rand('state', 0) %#ok<RAND>
            u_a_guess = rand(N,1)*2-1;
            u_a = u_a_guess + 1.0;
            itern = 0;
            while not(self.iter_terminate(u_a_guess, u_a, itern, self.opt.tolerance))
                % 0: Richardson's acceleration, or any numerical acceleration
                %    method for PLS where there is slow convergence?
                
                % Progress for PLS converges logarithmically from whatever
                % starting tolerance to the final tolerance.  Use a linear
                % mapping between 0 and 1, where 1 is mapped to log(tol).
                if itern == 3
                    start_perc = log(norm(u_a_guess - u_a));
                    final_perc = log(self.opt.tolerance);
                    progress_slope = (1-0)/(final_perc-start_perc);
                    progress_intercept = 0 - start_perc*progress_slope;
                end
                
                if self.opt.show_progress && itern > 2
                    perc = log(norm(u_a_guess - u_a))*progress_slope + progress_intercept;
                    stop_early = awaitbar(perc, h);
                    if stop_early
                        break;
                    end
                end
                
                % 0: starting point for convergence checking on next loop
                u_a_guess = u_a;
                
                % 1: Regress the score, u_a, onto every column in X, compute the
                %    regression coefficient and store in w_a
                % w_a = X.T * u_a / (u_a.T * u_a)
                w_a = regress_func(X, u_a, has_missing);
                
                % 2: Normalize w_a to unit length
                w_a = w_a / norm(w_a);
                
                % TODO(KGD): later on: investage NOT deflating X
                
                % 3: Now regress each row in X on the w_a vector, and store the
                %    regression coefficient in t_a
                % t_a = X * w_a / (w_a.T * w_a)
                t_a = regress_func(X, w_a, has_missing);

                % 4: Now regress score, t_a, onto every column in Y, compute the
                %    regression coefficient and store in c_a
                % c_a = Y.T * t_a / (t_a.T * t_a)
                c_a = regress_func(Y, t_a, has_missing);
                
                % 5: Now regress each row in Y on the c_a vector, and store the
                %    regression coefficient in u_a
                % u_a = Y * c_a / (c_a.T * c_a)
                %
                % TODO(KGD):  % Still handle case when entire row in Y is missing
                u_a = regress_func(Y, c_a, has_missing);
                
                itern = itern + 1;
            end
            
            % 6: To deflate the X-matrix we need to calculate the
            % loadings for the X-space.  Regress columns of t_a onto each
            % column in X and calculate loadings, p_a.  Use this p_a to
            % deflate afterwards.
            % Note the similarity with step 4! and that similarity helps
            % understand the deflation process.
            p_a = regress_func(X, t_a, has_missing); 

            if self.opt.show_progress
                if ishghandle(h)
                    close(h);
                end
            end
            
        end
    end % end methods (static)
    
end % end classdef