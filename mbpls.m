% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Multi-block PLS models

% Derived from the general class of "multiblock latent variable models", mblvm
classdef mbpls < mblvm
    properties (SetAccess = protected)
        Y = [];
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
            
        end % ``mbpls``        
        
        % Superclass abstract method implementation
        function self = expand_storage(self, varargin)
            % Do nothing: super-class methods are good enough
        end % ``expand_storage``
        
        % Superclass abstract method implementation
        function self = calc_model(self, A)
            % Fits a PLS latent variable model.
            %  
            % We assume the data are merged and preprocessed already.
            % Must also calculate all summary statistics for each block.
            
            % Perform ordinary missing data PLS on the merged block of data
            if numel(self.Y.mmap) > 1 && any(sum(self.Y.mmap, 2) == 0)
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
                    
                    ssq_before_Y = ssq(self.Y.data, 1);
                else
                    ssq_before = ssq(self.data, 1);
                    ssq_before_Y = ssq(self.Y.data, 1);
                end
                
                if all(ssq_before < self.opt.tolerance)
                    warning('mbpls:calc_model', 'There is no variance left in the X-data')
                end
                if all(ssq_before_Y < self.opt.tolerance)
                    warning('mbpls:calc_model', 'There is no variance left in the Y-data')
                end 
                                 
                % Converge onto a single component
                [t_a, p_a, w_a, c_a, u_a, itern] = mbpls.single_block_PLS(self.data, self.Y.data, self, a, self.has_missing); 
                
                % Flip the signs of the column vectors in P so that the largest
                % magnitude element is positive.
                % (Wold, Esbensen, Geladi, PCA, CILS, 1987, p 42
                %  http://dx.doi.org/10.1016/0169-7439(87)80084-9)
                [max_el, max_el_idx] = max(abs(w_a)); %#ok<ASGLU>
                if sign(w_a(max_el_idx)) < 1
                    p_a = -1.0 * p_a;
                    t_a = -1.0 * t_a;
                    w_a = -1.0 * w_a;
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
                        VIP_temp = VIP_temp + self.P{b}(:,a_iter) .^ 2 * self.stats{b}.VIP_f{a_iter,a} * self.K(b);
                    end
                    self.stats{b}.VIP_a(:,a) = sqrt(VIP_temp);
                
                    % Block T2 using the number of components calculated so far
                    self.stats{b}.T2(:,a) = self.mahalanobis_distance(self.T{b}(:,1:a));
                end
                w_super = regress_func(t_superblock, u_a, false);
                w_super = w_super / norm(w_super);
                
                % Store the super-level results
                self.super.T_summary(:,:,a) = t_superblock;
                self.super.T(:,a) = t_a;
                self.super.W(:,a) = w_super;
                                
                % Now deflate the data matrix using the superscore
                self.data = self.data - t_a * p_a';
                self.Y.data = self.Y.data - t_a * c_a';
                
                ssq_cumul = 0;
                ssq_before = 0;
                for b = 1:self.B
                    idx = self.b_iter(b);
                    X_portion = self.data(:, idx);
                    col_ssq = ssq(X_portion, 1)';
                    row_ssq = ssq(X_portion, 2);
                    ssq_cumul = ssq_cumul + sum(col_ssq);
                    ssq_before = ssq_before + sum(self.stats{b}.start_SS_col);
                    
                    self.stats{b}.R2k_a(:,a) = 1 - col_ssq ./ self.stats{b}.start_SS_col';
                    self.stats{b}.R2b_a(1,a) = 1 - sum(col_ssq) / sum(self.stats{b}.start_SS_col);
                    self.stats{b}.SSQ_exp(1,a) = sum(col_ssq);
                    if a>1
                        self.stats{b}.R2k_a(:,a) = self.stats{b}.R2k_a(:,a) - self.stats{b}.R2k_a(:,a-1);
                        self.stats{b}.R2b_a(1,a) = self.stats{b}.R2b_a(1,a) - self.stats{b}.R2b_a(1,a-1);
                    end
                    
                    self.stats{b}.SPE(:,a) = sqrt(row_ssq ./ numel(idx));
                end
                
                
                % Calculate the limits                
                self.calc_statistics_and_limits(a);
                
                self.A = a;
            end % looping on ``a`` latent variables
          
            
        end % ``calc_model``
    
        % Superclass abstract method implementation
        function state = apply_model(self, new, state, varargin) 
            
        end % ``apply_model``
        
        % Superclass abstract method implementation
        function self = calc_statistics_and_limits(self, a)
           
        end % ``calc_statistics_and_limits``
        
        % Superclass abstract method implementation
        function summary(self)
            
        end
        
    end % end methods (ordinary)
    
    % These methods don't require a class instance
    methods(Static)
        function [t_a, p_a, w_a, c_a, u_a, itern] = single_block_PLS(X, Y, self, a, has_missing)
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