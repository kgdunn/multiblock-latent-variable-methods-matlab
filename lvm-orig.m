classdef lvm 

    methods
        
        
        
        
        
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
        
        function [self, blockX, blockY] = fit_PLS(self, blockX, blockY, A)
            
            
            
            

            for a = which_components
                
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
        
            
    end % methods
end % end classdef

%-------- Helper functions (usually used in 2 or more places). May NOT use ``self``
