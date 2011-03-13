% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Multi-block PLS models

% Derived from the general class of "multiblock latent variable models", mblvm
classdef mbpls < mblvm
    properties (SetAccess = protected)
        Y = [];
        Y_has_missing = false;
    end
        
    methods 
        function self = mbpls(varargin)
            self = self@mblvm(varargin{:});            
        end % ``mbpls``        
        
        % Superclass abstract method implementation
        function self = expand_storage(self, varargin)
            % Do nothing: super-class methods are good enough
        end % ``expand_storage``
        
        % Superclass abstract method implementation
        function self = merge_blocks(self)
            
            % We will call the PCA class to do the work for us, after removing
            % the Y-block out to a new variable
            
            for b = 1:self.B
                if strcmpi(self.blocks{b}.name, 'y')
                    self.Y = self.blocks{b};
                    self.Y_has_missing = self.Y.has_missing;
                    self.blocks(b) = []; % delete the Y-block
                    
                    Ny = size(self.Y, 1);
                    if self.N ~= Ny
                        warning('mbpls:merge_blocks', 'The number of observations in all blocks must match.')
                    end
                end
            end
            
            merge_blocks@mblvm(self);
            
        end % ``merge_blocks``
        
        % Superclass abstract method implementation
        function self = calc_model(self, A)
            % Fits a PLS latent variable model.
            %  
            % We assume the data are merged and preprocessed already.
            % Must also calculate all summary statistics for each block.
            
            % Perform ordinary missing data PLS on the merged block of data
            if self.Y_has_missing
                if any(sum(self.Y.mmap, 2) == 0)
                    warning('mbpls:calc_model', ...
                            ['Cannot handle the case yet where the entire '...
                             'observation in Y-matrix is missing.  Please '...
                             'remove those rows and refit model.'])
                end
            end
            
            which_components = max(self.A+1, 1) : A;            
            for a = which_components
                
                start_time = cputime;
                % Baseline for all R2 calculations and variance check
                if a == 1             
                    ssq_before = ssq(self.data, 1);
                    self.split_result(ssq_before, 'stats', 'start_SS_col');
                    
                    ssq_before_Y = ssq(self.Y, 1);
                    self.stats.start_SS_col_Y = ssq_before_Y;
                    
                else
                    ssq_before = ssq(self.data, 1);
                    ssq_before_Y = ssq(self.Y, 1);
                end
                
                if all(ssq_before < self.opt.tolerance)
                    warning('mbpls:calc_model', 'There is no variance left in the X-data')
                end
                if all(ssq_before_Y < self.opt.tolerance)
                    warning('mbpls:calc_model', 'There is no variance left in the Y-data')
                end
                                
                % Converge onto a single component
                [t_a, p_a, w_a, c_a, u_a, itern] = mbpls.single_block_PLS(self.data, self.Y, self, a, self.has_missing); 
                
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
                
                % Now deflate the data matrix using the superscore
                
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
            p_a = regress_func(X, t_a, blockX.has_missing); 

            if self.opt.show_progress
                if ishghandle(h)
                    close(h);
                end
            end
            
        end
    end % end methods (static)
    
end % end classdef