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
        function self = expand_storage(self, A)
            % Do nothing: super-class methods are good enough
        end % ``expand_storage``
        
        function self = calc_model(self, A)
            % Fits a multiblock PCA model on the data, extracting A components
            % We assume the data are preprocessed already.
            % 
            % Must also calculate all summary statistics for each block.
            
            block_scaling = 1 ./ sqrt(self.K);
            if self.B == 1
                block_scaling = 1;
            end
                        
            X_merged = ones(self.N, sum(self.K)) .* NaN;
            has_missing = false;
            
            for b = 1:self.B
                X_merged(:, self.b_iter(b)) = self.blocks{b}.data .* block_scaling(b);
                if self.blocks{b}.has_missing
                    has_missing = true;
                end
            end

            % Perform ordinary missing data PCA on the merged block of data
            which_components = max(self.A+1, 1) : A;            
            for a = which_components
                
                start_time = cputime;
                % Baseline for all R2 calculations and variance check
                if a == 1             
                    ssq_before = ssq(X_merged, 1);
                    self.split_result(ssq_before, 'stats', 'start_SS_col');                    
                else
                    ssq_before = ssq(X_merged, 1);
                end
                
                if all(ssq_before < self.opt.tolerance)
                    warning('lvm:fit_PCA', 'There is no variance left in the data')
                end
                
                % Converge onto a single component
                [t_a, p_a, itern] = mbpca.single_block_PCA(X_merged, self, a, has_missing);                
                
                % Flip the signs of the column vectors in P so that the largest
                % magnitude element is positive.
                % (Wold, Esbensen, Geladi, PCA, CILS, 1987, p 42
                %  http://dx.doi.org/10.1016/0169-7439(87)80084-9)
                [max_el, max_el_idx] = max(abs(p_a));
                if sign(p_a(max_el_idx)) < 1
                    p_a = -1.0 * p_a;
                    t_a = -1.0 * t_a;
                end    
                
                self.model.stats.timing(a) = cputime - start_time;
                self.model.stats.itern(a) = itern;
                
                % Recover block information and store that.
                % TODO(KGD): optimize so we don't repeat this for single block                
                t_superblock = zeros(self.N, self.B);
                ssq_cumul = 0;
                for b = 1:self.B
                    idx = self.b_iter(b);
                    X_portion  = X_merged(:, idx);
                    
                    % Regress sub-columns of X_merged onto the superscore
                    % to get the block loadings.
                    p_b = regress_func(X_portion, t_a, has_missing);
                    
                    p_b = p_b / norm(p_b);
                    
                    % Block scores: regress rows of X onto the block loadings
                    t_b = regress_func(X_portion, p_b, has_missing);
                    
                    t_superblock(:,b) = t_b;
                    
                    % Store the block scores and loadings
                    self.T{b}(:,a) = t_b;
                    self.P{b}(:,a) = p_b;
                    
                    % Store statistics for each block.  The part we explain by
                    % this component is due to the *superscore*, t_a, not the
                    % blockscore.
                    X_portion_hat = t_a * p_b';
                    col_ssq = ssq(X_portion_hat, 1)';
                    row_ssq = ssq(X_portion_hat, 2);
                    ssq_cumul = ssq_cumul + sum(row_ssq);
                    self.stats{b}.R2k_a(:,a) = col_ssq ./ ssq_before(1, idx)';                    
                    self.stats{b}.R2b_a(1,a) = sum(row_ssq) / sum(ssq_before(1, idx));
                    
                    
                    ssq_after = ssq(X_portion - X_portion_hat, 2);
                    self.stats{b}.SPE(:,a) = sqrt(ssq_after ./ numel(idx));
                    VIP_temp = zeros(self.K(b), 1);
                    for a_iter = 1:a
                        self.stats{b}.VIP_f{a_iter,a} = sum(col_ssq) / (sum(ssq_before(1, idx)) - sum(ssq_after));
                        VIP_temp = VIP_temp + p_b .^ 2 * self.stats{b}.VIP_f{a_iter,a} * self.K(b);
                    end
                    self.stats{b}.VIP_a(:,a) = sqrt(VIP_temp);
                    
                    self.stats{b}.T2(:,a) = self.mahalanobis_distance(self.T{b}(:,1:a));
                end
                p_super = regress_func(t_superblock, t_a, false);
                     
                
                self.super.T(:,a) = t_a;
                self.super.P(:,a) = p_super;
                
                % Now deflate the data matrix
                X_merged = X_merged - t_a * p_a';
                
                % Cumulative R2 value for the whole component
                self.super.stats.R2(a) = ssq_cumul/sum(ssq_before);
                
                % Model summary SPE (not the superblock's SPE!), merely the
                % overall SPE from the merged model
                self.super.stats.SPE(:,a) = sqrt(ssq(X_merged, 2) ./ sum(self.K));
                
                % Model summary T2 (not the superblock's T2!), merely the
                % overall T2 from the merged model
                self.super.stats.T2(:,a) = self.mahalanobis_distance(self.super.T(:,1:a));
                
                self.super.stats.VIP_f
                self.super.stats.VIP
                
                % Calculate the limits                
                self.calc_statistics_and_limits(a);
                
                self.A = a;
                
            end % looping on ``a`` latent variables
        end % ``calc_model``
    
        function self = calc_statistics_and_limits(self, a)
            % Calculate summary statistics for the model. Given:
            % ``dblock``: the deflated block of data
            %
            % TODO
            % ----
            % * Modelling power of each variable
            % * Eigenvalues (still to come)
            
            % Check on maximum number of iterations
            if any(self.model.stats.itern >= self.opt.max_iter)
                warn_string = ['The maximum number of iterations was reached ' ...
                    'when calculating the latent variable(s). Please ' ...
                    'check the raw data - is it correct? and also ' ...
                    'adjust the number of iterations in the options.'];
                warning('lvm:calculate_statistics', warn_string)
            end
           
            % Calculate the limits for the latent variable model.
            %
            % References
            % ----------
            % [1]  SPE limits: Nomikos and MacGregor, Multivariate SPC Charts for
            %      Monitoring Batch Processes. Technometrics, 37, 41-59, 1995.
            %
            % [2]  T2 limits: Johnstone and Wischern.
            %
            % [3]  Score limits: two methods
            %
            %      A: Assume that scores are from a two-sided t-distribution with N-1
            %         degrees of freedom.  Based on the central limit theorem
            %
            %      B: (t_a/s_a)^2 ~ F_alpha(1, N-1) distribution if scores are
            %         assumed to be normally distributed, and s_a is chi-squared
            %         variable with N-1 DOF.
            %
            %         critical F = scipy.stats.f.ppf(0.95, 1, N-1)
            %         which happens to be equal to (scipy.stats.t.ppf(0.975, N-1))^2,
            %         as expected.  Therefore the alpha limit for t_a is equal to
            %         sqrt(scipy.stats.f.ppf(0.95, 1, N-1)) * S[:,a]
            %
            %      Both methods give the same limits. In fact, some previous code was:
            %         t_ppf_95 = scipy.stats.t.ppf(0.975, N-1)
            %         S[:,a] = std(this_lv, ddof=0, axis=0)
            %         lim.t['95.0'][a, :] = t_ppf_95 * S[:,a]
            %
            %      which assumes the scores were t-distributed.  In fact, the scores
            %      are not t-distributed, only the (score_a/s_a) is t-distributed, and
            %      the scores are NORMALLY distributed.
            %      S[:,a] = std(this_lv, ddof=0, axis=0)
            %      lim.t['95.0'][a, :] = n_ppf_95 * S[:,a]
            %      lim.t['99.0'][a, :] = n_ppf_99 * [:,a]
            %
            %      From the CLT: we divide by N, not N-1, but stddev is calculated
            %      with the N-1 divisor.
            
            siglevel = 0.95;
            
            for b = 1:self.B
                self.lim{1}.SPE = self.spe_limits(self.stats{1}.SPE, siglevel, self.K(b));
                self.super.lim.SPE = self.spe_limits(self. SPE, siglevel, self.K(b));
                
                
                
                %for siglevel_str in block.lim.T2.keys():
                %siglevel = float(siglevel_str)/100
                mult = a*(self.N-1)*(self.N+1)/(self.N*(self.N-a));
                limit = my_finv(siglevel, a, self.N-(a));
                block.lim.T2(:,a) = mult * limit;
                %end

                %for siglevel_str in block.lim.t.keys():
                alpha = (1-siglevel)/2.0;
                n_ppf = my_norminv(1-alpha);
                block.lim.t(:,a) = n_ppf * block.S(a);
                %end
            end
            
            
        end % ``calc_statistics_and_limits``
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
            rand('state', 0)
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

