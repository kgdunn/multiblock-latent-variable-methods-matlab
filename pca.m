% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for a single-block PCA model


% Create a handle class: only one instance of the class exists
classdef pca < mblvm
    methods 
        function self = pca(varargin)
            self = self@mblvm(varargin);            
        end % ``pca``
        
        
        
        % Superclass abstract method implementation
        function self = calc_model(self, A)
            % Fits a PCA model
            %
            % 1.   Wold, Esbensen and Geladi, 1987, Principal Component Analysis,
            %      Chemometrics and Intelligent Laboratory Systems, v 2, p37-52.
            %      http://dx.doi.org/10.1016/0169-7439(87)80084-9
            % 2.   Missing data: http://dx.doi.org/10.1016/B978-044452701-1.00125-3

            
            tolerance = self.opt.tolerance;            
            X = block.data;
            [N, K] = size(block.data);
            
            which_components = max(block.A+1, 1) : A;            
            for a = which_components  
                if self.opt.show_progress
                    h = awaitbar(0, sprintf('Calculating PCA component %d on block %s', a, block.name));
                end
                itern = 0;
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
                        warning('lvm:fit_PCA', 'There is no variance left in the data')
                end

                % Initialize t_a with random numbers, or carefully select a column
                % from X
%                 
%                 if a== 1
%                     % Which column had the largest variance initially?
%                     [s, minidx] = min(block.PP.scaling);        
%                 else
%                     % Find the column with largest variance
%                     [s, minidx] = max(nanstd(X));
%                 end
%                 t_a = X(:,minidx);
%                 % Does that column have missing values?
%                 t_a(isnan(t_a)) = 0.0;


                rand('state', 0)
                t_a_guess = rand(N,1)*2-1;
                t_a = t_a_guess + 1.0;

                while not(should_terminate(t_a_guess, t_a, itern, tolerance, self))
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
                    p_a = regress_func(X, t_a, block.has_missing);

                    % 2: Normalize p_a to unit length
                    p_a = p_a / sqrt(ssq(p_a));

                    % 3: Now regress each row in X on the p_a vector, and store the
                    %    regression coefficient in t_a
                    %t_a = X * p_a / (p_a.T * p_a)
                    %t_a = (X)(p_a) / ((p_a.T)(p_a))
                    %t_a = dot(X, p_a) / ssq(p_a)
                    t_a = regress_func(X, p_a, block.has_missing);

                    itern = itern + 1;
                end
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
                    block.P(:,a) = -1.0 * p_a;
                    block.T(:,a) = -1.0 * t_a;
                else
                    block.P(:,a) = p_a;
                    block.T(:,a) = t_a;
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
                    self = PCA_randomization_model(self, block);                    
                end
                

                self.stats.timing(a) = toc;
                self.stats.itern(a) = itern;

                % Loop terminated!  Now deflate the X-matrix
                X = X - t_a * p_a';
                % These are the Residual Sums of Squares (RSS); i.e X-X_hat
                row_SSX = ssq(X, 2);   % sum of squares along the row
                col_SSX = ssq(X, 1);   % sum of squares down the column

                block.stats.SPE(:,a) = sqrt(row_SSX/K);
                block.stats.deflated_SS_col(:,a) = col_SSX(:);
                block.stats.R2k_cum(:,a) = 1 - col_SSX./start_SS_col;

                % Cumulative R2 value for the whole block
                block.stats.R2(a) = 1 - sum(row_SSX)/sum(start_SS_col);
                

                % VIP value (only calculated for X-blocks); only last column is useful
                
                block.A = a;
                
            end % looping on ``a`` latent variables
            
            
            block.data = X; % Write the deflated array back to the block
            
        end % ``calc_model``
    end % methods
    
    
end % end classdef

