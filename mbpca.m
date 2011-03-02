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
        function self = calc_model(self, A)
            % Fits a multiblock PCA model on the data, extracting A components
            % We assume the data are preprocessed already.
            
            % FUTURE(expand to MBPCA)
            block_id = 1;

            start_time = cputime;
            dblock = self.blocks{block_id};
            which_components = max(self.A+1, 1) : A;            
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
                
                
                if self.opt.show_progress
                    if ishghandle(h)
                        close(h);
                    end
                end
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

                self.stats.timing(a) = cputime - start_time;
                self.stats.itern(a) = itern;

                % Loop terminated!  Now deflate the data matrix
                dblock.data = dblock.data - t_a * p_a';
                % These are the Residual Sums of Squares (RSS); i.e X - X_hat
                row_SSX = ssq(dblock.data, 2); % sum of squares along the row
                col_SSX = ssq(dblock.data, 1); % sum of squares down the column

                self.stats{block_id}.SPE(:,a) = sqrt(row_SSX/K);
                self.stats{block_id}.deflated_SS_col(:,a) = col_SSX(:);
                self.stats{block_id}.R2k_cum(:,a) = 1 - col_SSX./start_SS_col;

                % Cumulative R2 value for the whole block
                block.stats.R2(a) = 1 - sum(row_SSX)/sum(start_SS_col);
                

                % VIP value (only calculated for X-blocks); only last column is useful
                
                block.A = a;
                
            end % looping on ``a`` latent variables
            
            
            
        end % ``calc_model``
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

        end
    end % methods (static)
    
end % end classdef

