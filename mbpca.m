% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Multi-block PCA models

% Derived from the general class of "multiblock latent variable models", mblvm
classdef mbpca < mblvm
    
    methods 
        function self = mbpca(varargin)
            self = self@mblvm(varargin{:});            
        end % ``mbpca``        
        
        % Superclass abstract method implementation
        function self = expand_storage(self, varargin)
            % Do nothing: super-class methods are good enough
        end % ``expand_storage``
        
        % Superclass abstract method implementation
        function self = preprocess_extra(self)
        end
        
        % Superclass abstract method implementation
        function self = calc_model(self, A)
            % Fits a multiblock PCA model on the data, extracting A components
            % We assume the data are merged and preprocessed already.
            % 
            % Must also calculate all summary statistics for each block.
            
            % Perform ordinary missing data PCA on the merged block of data
            which_components = max(self.A+1, 1) : A;            
            for a = which_components
                
                start_time = cputime;
                % Baseline for all R2 calculations and variance check
                if a == 1             
                    ssq_before = ssq(self.data, 1);
                    self.split_result(ssq_before, 'stats', 'start_SS_col');                    
                else
                    ssq_before = ssq(self.data, 1);
                end
                
                if all(ssq_before < self.opt.tolerance)
                    warning('mbpca:calc_model', 'There is no variance left in the data')
                end
                
                % Converge onto a single component
                [t_a, p_a, itern] = mbpca.single_block_PCA(self.data, self, a, self.has_missing);                
                
                % Flip the signs of the column vectors in P so that the largest
                % magnitude element is positive.
                % (Wold, Esbensen, Geladi, PCA, CILS, 1987, p 42
                %  http://dx.doi.org/10.1016/0169-7439(87)80084-9)
                % 
                % Also see: http://www.models.life.ku.dk/signflipsvd
                % See 10.1.1.120.8476.pdf in the readings directory
                
                [max_el, max_el_idx] = max(abs(p_a)); %#ok<ASGLU>
                if sign(p_a(max_el_idx)) < 1
                    p_a = -1.0 * p_a;
                    t_a = -1.0 * t_a;
                end    
                
                self.model.stats.timing(a) = cputime - start_time;
                self.model.stats.itern(a) = itern;
                
                % Recover block information and store that.
                t_superblock = zeros(self.N, self.B);
                for b = 1:self.B
                    idx = self.b_iter(b);
                    X_portion  = self.data(:, idx);
                    
                    % Regress sub-columns of self.data onto the superscore
                    % to get the block loadings.
                    p_b = regress_func(X_portion, t_a, self.has_missing);
                    
                    p_b = p_b / norm(p_b);
                    
                    % Block scores: regress rows of X onto the block loadings
                    t_b = regress_func(X_portion, p_b, self.has_missing);
                    
                    t_superblock(:,b) = t_b;
                    
                    % Store the block scores and loadings
                    self.T{b}(:,a) = t_b;
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
                        VIP_temp = VIP_temp + self.P{b}(:,a_iter) .^ 2 * self.stats{b}.VIP_f{a_iter,a} * self.K(b);
                    end
                    self.stats{b}.VIP_a(:,a) = sqrt(VIP_temp);
                
                    [self.stats{b}.T2(:,a), S] = self.mahalanobis_distance(self.T{b}(:,1:a));
                    self.stats{b}.S = S;
                end
                p_super = regress_func(t_superblock, t_a, false);
                     
                self.super.T_summary(:,:,a) = t_superblock;
                self.super.T(:,a) = t_a;
                self.super.P(:,a) = p_super;
                
                % Now deflate the data matrix using the superscore
                self.data = self.data - t_a * p_a';
                ssq_cumul = 0;
                ssq_before = 0;
                for b = 1:self.B
                    idx = self.b_iter(b);
                    X_portion = self.data(:, idx);
                    col_ssq = ssq(X_portion, 1)';
                    row_ssq = ssq(X_portion, 2);
                    ssq_cumul = ssq_cumul + sum(col_ssq);
                    ssq_before = ssq_before + sum(self.stats{b}.start_SS_col);
                    
                    self.stats{b}.R2Xk_a(:,a) = 1 - col_ssq ./ self.stats{b}.start_SS_col';
                    self.stats{b}.R2Xb_a(1,a) = 1 - sum(col_ssq) / sum(self.stats{b}.start_SS_col);
                    self.stats{b}.SSQ_exp(1,a) = sum(col_ssq);
                    if a>1
                        self.stats{b}.R2Xk_a(:,a) = self.stats{b}.R2Xk_a(:,a) - sum(self.stats{b}.R2Xk_a(:,1:a-1), 2);
                        self.stats{b}.R2Xb_a(1,a) = self.stats{b}.R2Xb_a(1,a) - sum(self.stats{b}.R2Xb_a(1,1:a-1), 2);
                    end
                    
                    self.stats{b}.SPE(:,a) = row_ssq;
                end
                
                % TODO(KGD): sort out R2, SPE  and VIP calculations for each
                % block. Do we use the block loading, or the overall loadings?
                    
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
                self.super.SPE(:,a) = row_ssq_deflated;
                
                % Model summary T2 (not the superblock's T2!), merely the
                % overall T2 from the merged model
                [self.super.T2(:,a), S] = self.mahalanobis_distance(self.super.T(:,1:a));
                self.super.S = S;
                
                VIP_temp = zeros(sum(self.B), 1);
                for a_iter = 1:a
                    self.super.stats.VIP_f{a_iter,a} = self.super.stats.SSQ_exp(1,a_iter) / sum(self.super.stats.SSQ_exp);
                    VIP_temp = VIP_temp + self.super.P(:,a_iter) .^ 2 * self.super.stats.VIP_f{a_iter,a} * sum(self.B);
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

        end % ``calc_model_post``
    
        % Superclass abstract method implementation
        function state = apply_model(self, new, state) 
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
                    state.T{b}(:,a) = regress_func(new{b}.data, self.P{b}(:,a), new{b}.has_missing);
                    state.T{b}(:,a) = state.T{b}(:,a) .* self.block_scaling(b);
                    % Transfer it to the superscore matrix
                    state.T_sb(:,b,a) = state.T{b}(:,a);
                end
                
                % Calculate the superscore, T_super. Verify that self.super.P
                % is always of unit length.
                state.T_super(:,a) = state.T_sb(:,:,a) * self.super.P(:,a);
                
                % Deflate each block: using the SUPERSCORE and the block loading
                for b = 1:self.B
                    deflate = state.T_super(:,a) * self.P{b}(:,a)';
                    state.stats.R2{b}(:,1) = ssq(deflate, 2) ./ state.stats.initial_ssq{b};
                    new{b}.data = new{b}.data - deflate;
                end            
            end % looping on ``a`` latent variables
            
            
            % Summary statistics for each block and the super level
            overall_variance = zeros(state.Nnew, 1);
            for b = 1:self.B
                block_variance = ssq(new{b}.data, 2);
                overall_variance = overall_variance + block_variance;
                state.stats.SPE{b} = block_variance;
            end
            state.stats.super.R2(:,1) = 1 - overall_variance ./state.stats.initial_ssq_total;
            
            state.stats.super.SPE(:,1) = overall_variance;
            state.stats.super.T2(:,1) = mblvm.mahalanobis_distance(state.T_super, self.super.S);
            
            
        end % ``apply_model``
        
        % Superclass abstract method implementation
        function summary(self)
            % Displays more information about ``self``
            
            % TODO(KGD): improve this by not displaying block info if B==1
            
            fprintf('R2 summary for %s (all as percentages)\n', self.model_type);
            w_char = zeros(self.B,1);
            ncols = 2 + self.B;  % A column, overall R2, and then all blocks            
            line_length = 4 + 8 + ncols;
            all_lines = '|%3i|%7.2f||';
            start_line = '|%3s|%7s||';
            block_names = cell(self.B,1);
            for b = 1:self.B
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
            all_lines = [all_lines, '\n'];
            start_line = [start_line, '\n'];
            
            disp(repmat('-', 1, line_length))
            fprintf_args = {' A ', ' Total ', block_names{:}};
            fprintf(start_line, fprintf_args{:});
            disp(repmat('-', 1, line_length))
            
            
            for a = 1:self.A
                fprintf_args = zeros(1, 2+self.B);
                fprintf_args(1:2) =  [a, self.super.stats.R2X(a)*100];
                for b = 1:self.B
                    fprintf_args(2+b) = self.stats{b}.R2Xb_a(a)*100;
                end
                fprintf(all_lines, fprintf_args);
            end
            
            disp(repmat('-', 1, line_length))
            fprintf('Overall R2X(cumul) = %6.2f%%\n', sum(self.super.stats.R2X)*100)
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
       
        % Superclass abstract method implementation
        function out = register_plots_post(self)
            out = [];
            
        end
        
        % Superclass abstract method implementation
        function stat = randomization_objective(self)
            score_vector = self.super.T(:, self.A);
            stat = sqrt(self.robust_scale(score_vector));
        end % ``randomization_objective``
        
        % Superclass abstract method implementation
        function randomization_test_launch(self)
            % Setup required before running the randomization permutations
            
            % Store the original Y. We will restore it afterwards
            %self.opt.randomize_test.temp_data = self.Y.data;
            
        end % ``randomization_test_launch``
        
        % Superclass abstract method implementation
        function randomization_test_finish(self)
            % Clean up after running the randomization permutations
            
            % Store the original Y. We will restore it afterwards
            %self.Y.data = self.opt.randomize_test.temp_data;            
        end % ``randomization_test_finish``
    
        % Superclass abstract method implementation
        function output = randomization_permute_and_build(self)
            % Function to permute the data and  build the model without error 
            % checks and delays
            %
            % NOTE: this function must not reset the random number generator.
            %       That will already be set ahead of time in the calling
            %       function.
            % 
            % Must return a structure, ``output`` that will be sent to 
            % ``self.randomization_objective(...)`` in ``varargin``.
            % So store in ``output`` all the entries required to evaluate
            % the randomization objective function.
            
                    
            % Calculate the "a"th component using this permuted Y-matrix, but
            % the unpermuted X-matrix.
            
            
        end % ``randomization_test_finish``
        
    end % end methods (ordinary)
    
    % These methods don't require a class instance
    methods(Static)
        function [t_a, p_a, itern] = single_block_PCA(dblock, self, a, has_missing)
            % Extracts a PCA component on a single block of data, ``data``.
            % The model object, ``self``, should also be provided, for options.
            % The ``a`` entry is merely used to show which component is being
            % extracted in the progress bar.
            % The ``has_missing`` flag is used to indicate if any entries in 
            % ``dblock`` are missing.
            %
            %
            % 1.   Wold, Esbensen and Geladi, 1987, Principal Component Analysis,
            %      Chemometrics and Intelligent Laboratory Systems, v 2, p37-52.
            %      http://dx.doi.org/10.1016/0169-7439(87)80084-9
            % 2.   Missing data: http://dx.doi.org/10.1016/B978-044452701-1.00125-3
            
            tolerance = self.opt.tolerance;
            N = size(dblock, 1);
            if self.opt.show_progress
                h = awaitbar(0, sprintf('Calculating component %d', a));
            end
            rand('state', 0) %#ok<RAND>
            t_a_guess = rand(N,1)*2-1;
            t_a = t_a_guess + 1.0;
            itern = 0;
            while not(self.iter_terminate(t_a_guess, t_a, itern, tolerance))
                % 0: Richardson's acceleration, or any numerical acceleration
                %    method for PCA where there is slow convergence?
                
                % Progress for PCA converges logarithmically from whatever
                % starting tolerance to the final tolerance.  Use a linear
                % mapping between 0 and 1, where 1 is mapped to log(tol).
                if itern == 3
                    start_perc = log(norm(t_a_guess - t_a));
                    final_perc = log(tolerance);
                    progress_slope = (1-0)/(final_perc-start_perc);
                    progress_intercept = 0 - start_perc*progress_slope;
                end
                
                if self.opt.show_progress && itern > 2
                    perc = log(norm(t_a_guess - t_a))*progress_slope + progress_intercept;
                    stop_early = awaitbar(perc, h);
                    if stop_early
                        break;
                    end
                end
                
                % 0: starting point for convergence checking on next loop
                t_a_guess = t_a;
                
                % 1: Regress the score, t_a, onto every column in X, compute the
                %    regression coefficient and store in p_a
                %p_a = X.T * t_a / (t_a.T * t_a)
                %p_a = (X.T)(t_a) / ((t_a.T)(t_a))
                %p_a = dot(X.T, t_a) / ssq(t_a)
                p_a = regress_func(dblock, t_a, has_missing);
                
                % 2: Normalize p_a to unit length
                p_a = p_a / sqrt(ssq(p_a));
                
                % 3: Now regress each row in X on the p_a vector, and store the
                %    regression coefficient in t_a
                %t_a = X * p_a / (p_a.T * p_a)
                %t_a = (X)(p_a) / ((p_a.T)(p_a))
                %t_a = dot(X, p_a) / ssq(p_a)
                t_a = regress_func(dblock, p_a, has_missing);
                
                itern = itern + 1;
            end
            
            if self.opt.show_progress
                if ishghandle(h)
                    close(h);
                end
            end
            
        end
    end % end methods (static)
    
end % end classdef

