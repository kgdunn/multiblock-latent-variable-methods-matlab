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
            
            % FUTURE(expand to MBPCA)
            block_id = 1;
            
            start_time = cputime;
            dblock = self.blocks{block_id};
            which_components = max(self.A+1, 1) : A;
            
            K = size(dblock.data, 2);
            for a = which_components
                if a == 1                
                    dblock.stats.start_SS_col = ssq(dblock.data, 1);
                    initial_ssq = dblock.stats.start_SS_col;
                else
                    initial_ssq = ssq(dblock.data, 1);
                end
                % Baseline for all R^2 calculations
                start_SS_col = dblock.stats.start_SS_col;
                if all(initial_ssq < self.opt.tolerance)
                        warning('lvm:fit_PCA', 'There is no variance left in the data')
                end
                
                % Do the work elsewhere
                [t_a, p_a, itern] = mbpca.single_block_PCA(dblock, self, a);                
                
                % Store results
                % -------------
                % Flip the signs of the column vectors in P so that the largest
                % magnitude element is positive (Wold, Esbensen, Geladi, PCA,
                % CILS, 1987, p 42)
                [max_el, max_el_idx] = max(abs(p_a));
                if sign(p_a(max_el_idx)) < 1
                    self.P{block_id}(:,a) = -1.0 * p_a;
                    self.T{block_id}(:,a) = -1.0 * t_a;
                else
                    self.P{block_id}(:,a) = p_a;
                    self.T{block_id}(:,a) = t_a;
                end                
                
                % We should do our randomization test internal to the PLS
                % algorithm, before we deflate.
                if self.opt.randomize_test.quick_return
                    self = cell(1);
                    self{1}.T = zeros(N, A);
                    self{1}.T(:,a) = t_a;
                    self{1}.itern = itern;
                    return;
                end
                if self.opt.randomize_test.use_internal
                    self = PCA_randomization_model(self, block_id);                    
                end                

                self.model.stats.timing(a) = cputime - start_time;
                self.model.stats.itern(a) = itern;

                % Loop terminated!  Now deflate the data matrix
                dblock.data = dblock.data - t_a * p_a';
                
                % These are the Residual Sums of Squares (RSS); i.e X - X_hat
                row_SSX = ssq(dblock.data, 2); % sum of squares along the row
                col_SSX = ssq(dblock.data, 1); % sum of squares down the column

                self.stats{block_id}.SPE(:,a) = sqrt(row_SSX/K);
                self.stats{block_id}.deflated_SS_col(:,a) = col_SSX(:);
                self.stats{block_id}.R2k_cum(:,a) = 1 - col_SSX./start_SS_col;

                % Cumulative R2 value for the whole block
                self.stats{block_id}.R2(a) = 1 - sum(row_SSX)/sum(start_SS_col);
                

                % VIP value (only calculated for X-blocks); only last column is useful
                
                 self.A = a;
                
            end % looping on ``a`` latent variables
        end % ``calc_model``
    
        function self = calc_statistics_and_limits(self, varargin)
            % Calculate summary statistics for the model.
            %
            % TODO
            % ----
            % * Modelling power of each variable
            % * Eigenvalues (still to come)
            % * Squared prediction error and T2
            
            % If ``varargin`` is supplied, then we must calculate the
            % statistics on the ``varargin{1}``
            model = self;
            if nargin==2 && isa(varargin{1}, 'struct')
                testing_data = true;
                self = varargin{1};
            else
                testing_data = false;
                model = self;
            end
            
            for b = 1:model.B
                if self.A == 0
                    continue
                end
                
                
                %block = self.blocks{b};
                % Calculate the R2 explained on a per-component basis, for the block
                self.stats{b}.R2_a = [self.stats{b}.R2(1); diff(self.stats{b}.R2)];
                
                % Calculate R2 explained on a per-component basis, for each variable
                first_PC = self.stats{b}.R2k_cum(:,1);
                self.stats{b}.R2k_a = [first_PC, diff(self.stats{b}.R2k_cum, 1, 2)];
                
                % VIP values =  sqrt{ SSQ[ (P.^2) * (R2_per_LV), row_wise ] }
                % TODO(KGD): come back to VIP calculation and find a
                % reference for it
                
                % Check on maximum number of iterations
                if any(model.model.stats.itern >= model.opt.max_iter)
                    warn_string = ['The maximum number of iterations was reached ' ...
                        'when calculating the latent variable(s). Please ' ...
                        'check the raw data - is it correct? and also ' ...
                        'adjust the number of iterations in the options.'];
                    warning('lvm:calculate_statistics', warn_string)
                end
                
                % Variance of each latent variable score.
                self.S{b} = std(self.T{b}, 1);
                
%                 % Not calculated  for certain blocks: e.g. in PLS, the
%                 % block.T is empty.
%                 if size(self.T{b}, 2) == self.A
%                     for a = 1:self.A
%                         self.stats.T2(:,a) = sum((block.T(:,1:a) ./ ...
%                             repmat(block.S(:,1:a), block.N,1)).^2, 2);
%                     end
%                 end
%                 
%                 % TODO(KGD): Modelling power = 1 - (RSD_k)/(RSD_0k)
%                 % TODO(KGD): Is this valid/useful for Y-blocks?
%                 % Strictly speaking, RSD is a standard deviation.  We will use
%                 % sums of squares though, because the DoF are the same.
%                 %block.stats.model_power(1,:) = ...
%                 %       1.0 - sqrt(ssq(block.data, 1) ./ block.stats.start_SS_col);
%                 self.blocks{b} = block;
            end
            
            % Calculate the limits for the latent variable model.
            %
            % References
            % ----------
            % [1]  SPE limits: Nomikos and MacGregor, Multivariate SPC Charts for
            %      Monitoring Batch Processes. Technometrics, 37, 41-59, 1995.
            %
            % [2]  T2 limits: Johnstone and Wischern.
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
            for b = 1:self.B
                block = self.blocks{b};
                N = block.N;
                siglevel = 0.95;
                for a = 1:self.A
                    
                    SPE_values = block.stats.SPE(:,a);
                    var_SPE = var(SPE_values);
                    avg_SPE = mean(SPE_values);
                    chi2_mult = var_SPE/(2.0 * avg_SPE);
                    chi2_DOF = (2.0*avg_SPE^2)/var_SPE;
                    
                    %for siglevel_str in block.lim.SPE.keys():
                    %    siglevel = float(siglevel_str)/100
                    block.lim.SPE(:,a) = chi2_mult * my_chi2inv(siglevel, chi2_DOF);
                    
                    % For batch blocks: calculate instantaneous SPE using a window
                    % of width = 2w+1 (default value for w=2).
                    % This allows for (2w+1)*N observations to be used to calculate
                    % the SPE limit, instead of just the usual N observations.
                    %
                    % Also for batch systems:
                    % low values of chi2_DOF: large variability of only a few variables
                    % high values: more stable periods: all k's contribute
                    
                    %for siglevel_str in block.lim.T2.keys():
                    %siglevel = float(siglevel_str)/100
                    mult = a*(N-1)*(N+1)/(N*(N-a));
                    limit = my_finv(siglevel, a, N-(a));
                    block.lim.T2(:,a) = mult * limit;
                    %end
                    
                    %for siglevel_str in block.lim.t.keys():
                    alpha = (1-siglevel)/2.0;
                    n_ppf = my_norminv(1-alpha);
                    block.lim.t(:,a) = n_ppf * block.S(a);
                    %end
                end
                self.blocks{b} = block;
            end
            
            Then call the subclass methods
            
%             if self.opt.batch.calculate_monitoring_limits
%                 for b = 1:self.B
%                     if strcmp(self.blocks{b}.block_type, 'batch')
%                         
%                     end
%                 end
%             end
        end % ``calc_statistics_and_limits``
    end % methods
    
    % These methods don't require a class instance
    methods(Static)
        function [t_a, p_a, itern] = single_block_PCA(dblock, self, a)
            % Extracts a PCA component on a single block of data, ``data``.
            % The model object, ``self``, should also be provided, for options.
            % The ``a`` entry is merely used to show which component is being
            % extracted in the progress bar.
            %
            %
            % 1.   Wold, Esbensen and Geladi, 1987, Principal Component Analysis,
            %      Chemometrics and Intelligent Laboratory Systems, v 2, p37-52.
            %      http://dx.doi.org/10.1016/0169-7439(87)80084-9
            % 2.   Missing data: http://dx.doi.org/10.1016/B978-044452701-1.00125-3
            
            tolerance = self.opt.tolerance;
            N = size(dblock.data, 1);
            if self.opt.show_progress
                h = awaitbar(0, sprintf('Calculating component %d on block %s', a, dblock.name));
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
                p_a = regress_func(dblock.data, t_a, dblock.has_missing);
                
                % 2: Normalize p_a to unit length
                p_a = p_a / sqrt(ssq(p_a));
                
                % 3: Now regress each row in X on the p_a vector, and store the
                %    regression coefficient in t_a
                %t_a = X * p_a / (p_a.T * p_a)
                %t_a = (X)(p_a) / ((p_a.T)(p_a))
                %t_a = dot(X, p_a) / ssq(p_a)
                t_a = regress_func(dblock.data, p_a, dblock.has_missing);
                
                itern = itern + 1;
            end
            
            if self.opt.show_progress
                if ishghandle(h)
                    close(h);
                end
            end
            
        end
    end % methods (static)
    
end % end classdef

