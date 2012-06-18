% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% Licensed under the BSD license.
% -------------------------------------------------------------------------
%
% Multi-block PLS models

% Derived from the general class of "multiblock latent variable models", mblvm
classdef mbpls < mblvm
    properties (SetAccess = protected)
        Y = [];
        Y_copy = [];  % A copy of the Y-block before any calculations
        Y_hat = [];   % Predictions of the Y-block
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
            self.Y_copy = copy(Y_block);
            self.Y_hat = copy(Y_block);
            self.Y_hat.data = self.Y_hat.data .* 0;
            self.M = shape(self.Y, 2);
            self.has_missing = self.has_missing | self.Y.has_missing;
                         
        end % ``mbpls``        
        
        % Another painful example of MATLAB's poor OOP. Cannot redefine a 
        % getter method on a superclass variable, ``N`` in this case.
        %
        %function N = get.N(self)
        %    N = size(self.blocks{1}.data, 1);
        %    for b = 1:self.B
        %        assert(N == self.blocks{b}.N);
        %    end
        %    assert(N == self.Y.N)
        %end
        
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

        function add_next_component = add_next_component(self, requested_A, a, ssq_X, ssq_Y)
            % Certain conditions should be met, while other conditions must be
            % met in order to add the next component.            
            
            % These conditions *should* be met in order to add the next 
            % component:

            % Termination condition: fixed request for a certain number 
            % of components.
            request = a < requested_A;            
           
            % Termination condition: randomization test
            randomize_keep_adding = false;
            if self.opt.randomize_test.use
                randomize_keep_adding = not(self.randomization_should_terminate());
                if a ~= 0 && not(isempty(self.opt.randomize_test.risk_statistics{a}))
                    stats = self.opt.randomize_test.risk_statistics{a};
                    self.model.stats.risk.stats{a} = stats;
                    self.model.stats.risk.rate(a) = stats.num_G_exceeded/stats.nperm * 100;
                    self.model.stats.risk.objective(a) = self.opt.randomize_test.test_statistic(a);
                end
            end
            
            % Combine all conditions that *should* be met
            add_next_component = request || randomize_keep_adding;
            
            % We already know the result. Return early.
            if not(add_next_component)
                return
            end
            
            % These next conditions can veto the addition of another component            
            variance_left = true;
            if all(ssq_X < self.opt.tolerance)
                variance_left = false;
                warning('mbpls:calc_model:no_X_variance', ['There is no variance left ', ...
                        'in the X-data. A new component will not be added.'])
            end
            if all(ssq_Y < self.opt.tolerance)
                variance_left = false;
                warning('mbpls:calc_model:no_Y_variance', ['There is no variance left ', ...
                        'in the Y-data. A new component will not be added.'])
            end            
            veto = self.opt.stop_now || not(variance_left);            
            
            add_next_component = add_next_component && not(veto);
        end % ``add_next_component``
        
        % Superclass abstract method implementation
        function self = calc_model(self, requested_A)
            % Fits a PLS latent variable model.
            %  
            % We assume the data are merged and preprocessed already.
            % Must also calculate all summary statistics for each block.
            
            % First check that the number of samples is constant throughout 
            % all blocks.  ``self.N`` checks consistency between all X-blocks
            if self.N ~= self.Y.N
                error('mbpls:calc_model', 'Inconsistent number of observations in Y-block')
            end
            
            % Perform ordinary missing data PLS on the merged block of data
            if not(isempty(self.Y.mmap)) > 1 && any(sum(self.Y.mmap, 2) == 0)
                warning('mbpls:calc_model', ...
                        ['Cannot handle the case yet where the entire '...
                         'observation in Y-matrix is missing.  Please '...
                         'remove those rows and refit model.'])
            end            
                       
            a = max(self.A+1,1);
                    
            
            % Baseline for all R2 calculations and variance check
            ssq_X_before = self.ssq(self.data, 1);
            ssq_Y_before = self.ssq(self.Y.data, 1);
            if a == 1                
                self.split_result(ssq_X_before, 'stats', 'start_SS_col');
                self.super.stats.ssq_Y_before = ssq_Y_before;
            end
            
            add_next_component = self.add_next_component(requested_A, a-1, ...
                                ssq_X_before, ssq_Y_before);
            
            while add_next_component
                
                start_time = cputime;

                % Converge onto a single component
                if self.opt.show_progress
                    self.progressbar(sprintf('Calculating component %d', a))
                end
                out = self.single_block_PLS(self.data, self.Y.data);
                
                % Flip the signs of the column vectors in P so that the largest
                % magnitude element is positive.
                % (Wold, Esbensen, Geladi, PCA, CILS, 1987, p 42
                %  http://dx.doi.org/10.1016/0169-7439(87)80084-9)
                [max_el, max_el_idx] = max(abs(out.w_a)); %#ok<ASGLU>
                if sign(out.w_a(max_el_idx)) < 1
                    out.p_a = -1.0 * out.p_a;
                    out.t_a = -1.0 * out.t_a;
                    out.c_a = -1.0 * out.c_a;
                    out.u_a = -1.0 * out.u_a;
                    out.w_a = -1.0 * out.w_a;
                end  
                
                % Recover block information and store that.
                t_superblock = zeros(self.N, self.B);
                for b = 1:self.B
                    idx = self.b_iter(b);
                    X_portion  = self.data(:, idx);
                    
                    % Regress sub-columns of self.data onto the superscore
                    % to get the block weights.
                    w_b = self.regress_func(X_portion, out.u_a, self.has_missing);
                    
                    w_b = w_b / norm(w_b);
                    
                    % Block scores: regress rows of X onto the block weights
                    t_b = self.regress_func(X_portion, w_b, self.has_missing);
                    t_superblock(:,b) = t_b;
                    %T_b_recovered{b}(:,a) = X_portion * w_b / (w_b'*w_b) / sqrt(K_b(b));
                    
                    
                    % Block loadings: that would have been used to deflate the
                    % X-blocks
                    p_b = self.regress_func(X_portion, out.t_a, self.has_missing);
                    
                    % Store the block scores, weights and loadings
                    self.T{b}(:,a) = t_b;
                    self.W{b}(:,a) = w_b;
                    self.P{b}(:,a) = p_b;
                   
                    % Store the SS prior to deflation 
                    X_portion_hat = out.t_a * p_b';
                    self.stats{b}.col_ssq_prior(:, a) = self.ssq(X_portion_hat,1);
                    
                    % VIP calculations
                    % -----------------                    
                    ssq_after = self.ssq(X_portion - X_portion_hat, 1);
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
                w_super = self.regress_func(t_superblock, out.u_a, false);
                w_super = w_super / norm(w_super);
                
                % Store the super-level results
                self.super.T_summary(:,:,a) = t_superblock;
                self.super.T(:,a) = out.t_a;
                self.super.W(:,a) = w_super;
                self.super.C(:,a) = out.c_a;
                self.super.U(:,a) = out.u_a;
                
                
                % Randomization testing for this component
                if self.opt.randomize_test.use
                    self.randomization_test(a);
                end                
                                
                % Now deflate the data matrix using the superscore
                self.data = self.data - out.t_a * out.p_a';
                
                % Make current predictions of Y using all available PCs
                Y_hat_update = out.t_a * out.c_a';
                self.Y_hat.data = self.Y_hat.data + Y_hat_update;
                self.Y.data = self.Y.data - Y_hat_update;
                ssq_Y_after = self.ssq(self.Y.data, 1)';
                self.super.stats.R2Yk_a(:,a) = 1 - ssq_Y_after ./ self.super.stats.ssq_Y_before';
                self.super.stats.R2Y(a) = 1 - sum(ssq_Y_after)/ sum(self.super.stats.ssq_Y_before);
                if a>1
                    self.super.stats.R2Y(a) = self.super.stats.R2Y(a) - sum(self.super.stats.R2Y(1:a-1), 2);
                    self.super.stats.R2Yk_a(:,a) = self.super.stats.R2Yk_a(:,a) - sum(self.super.stats.R2Yk_a(:,1:a-1), 2);
                end
                
                ssq_cumul = 0;
                ssq_X_before = 0;
                for b = 1:self.B
                    idx = self.b_iter(b);
                    
                    % X_portion has already been deflated by the current PC
                    X_portion = self.data(:, idx);
                    
                    % Calculate SPE
                    self.stats{b}.SPE(:,a) = self.ssq(X_portion, 2);
                    
                    % Calculate R2 per variable
                    col_ssq_remain = self.ssq(X_portion, 1)';
                    ssq_cumul = ssq_cumul + sum(col_ssq_remain);
                    ssq_X_before = ssq_X_before + sum(self.stats{b}.start_SS_col);
                    
                    self.stats{b}.R2Xk_a(:,a) = 1 - col_ssq_remain ./ self.stats{b}.start_SS_col';
                    self.stats{b}.R2Xb_a(1,a) = 1 - sum(col_ssq_remain) / sum(self.stats{b}.start_SS_col);
                    self.stats{b}.SSQ_exp(1,a) = sum(col_ssq_remain);
                    if a>1
                        self.stats{b}.R2Xk_a(:,a) = self.stats{b}.R2Xk_a(:,a) - sum(self.stats{b}.R2Xk_a(:,1:a-1), 2);
                        self.stats{b}.R2Xb_a(1,a) = self.stats{b}.R2Xb_a(1,a) - sum(self.stats{b}.R2Xb_a(1,1:a-1), 2);
                    end
                end                
                
                % Cumulative R2 value for the whole component
                self.super.stats.R2X(a) = 1 - ssq_cumul/ssq_X_before;
                if a>1
                    self.super.stats.R2X(a) = self.super.stats.R2X(a) - sum(self.super.stats.R2X(1:a-1), 2);
                end
                
                % Store explained variance
                self.super.stats.SSQ_exp(1,a) = ssq_cumul;
                
                % Model summary SPE (not the superblock's SPE!), merely the
                % overall SPE from the merged model
                self.super.SPE(:,a) = self.ssq(self.data, 2);
                
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
                
                
                self.model.stats.timing(a) = cputime - start_time;
                self.model.stats.itern(a) = out.itern;
                
                self.A = a;  
                
                % Do we add another component?
                add_next_component = self.add_next_component(requested_A, a, col_ssq_remain, ssq_Y_after);
                
                a = a + 1;
                
                
            end % looping on ``a`` latent variables

            if self.opt.randomize_test.use                
                self.A = self.opt.randomize_test.last_worthwhile_A;
            end
                      
        end % ``calc_model``
        
        % Superclass abstract method implementation
        function stat = randomization_objective(self, current_A, varargin)
             % 12 May 2011: Use Matthews correlation coefficient for categorical,
             % single Y-variables.
             %
             % * http://en.wikipedia.org/wiki/Matthews_correlation_coefficient
             % * http://en.wikipedia.org/wiki/Receiver_operating_characteristic
             
             if nargin == 2
                overall_T = self.super.T(:,current_A);
                overall_U = self.super.U(:,current_A);
             else
                 overall_T = varargin{1}.t_a;
                 overall_U = varargin{1}.u_a;
             end
                
             stat = (overall_U' * overall_T) ./ ...
                 (sqrt(overall_T'*overall_T) .* sqrt(overall_U' * overall_U));
            if isnan(stat)
                stat = 0.0;
            end
            
        end % ``randomization_objective``
        
        % Superclass abstract method implementation
        function randomization_test_launch(self)
            % Setup required before running the randomization permutations
            
            % Store the original Y. We will restore it afterwards
            self.opt.randomize_test.temp_data = self.Y.data;
            
        end % ``randomization_test_launch``
        
        % Superclass abstract method implementation
        function randomization_test_finish(self)
            % Clean up after running the randomization permutations
            
            % Store the original Y. We will restore it afterwards
            self.Y.data = self.opt.randomize_test.temp_data;            
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
            self.Y.data = self.Y.data(randperm(self.N), :);
                    
            % Calculate the "a"th component using this permuted Y-matrix, but
            % the unpermuted X-matrix.
            output = self.single_block_PLS(self.data, self.Y.data);
            
        end % ``randomization_test_finish``
    
        % Superclass abstract method implementation
        function limits_subclass(self)
            % Calculates the monitoring limits for a batch blocks in the model
            for b = 1:self.B                
            end
            
        end % ``calc_model_post``
        
        % Superclass abstract method implementation
        function state = apply_model(self, new, state, varargin) 
            % Applies a PLS model to the given ``block`` of (new) data.
            % 
            % TODO(KGD): allow user to specify ``A``
            % TODO(KGD): check that this matches the actual model for the
            % block scores still
            
            which_components = 1 : min(self.A);
            for a = which_components
                
                initial_ssq_total = zeros(state.Nnew, 1);
                initial_ssq = cell(1, self.B);
                for b = 1:self.B           
                    if not(isempty(new{b}))
                        initial_ssq{b} = self.ssq(new{b}.data, 2);
                        initial_ssq_total = initial_ssq_total + initial_ssq{b};
                        if a==1                        
                            state.stats.initial_ssq{b} = initial_ssq{b};
                            state.stats.initial_ssq_total(:,1) = initial_ssq_total;
                        end
                    end
                end

                if all(initial_ssq_total < self.opt.tolerance)
                    warning('mbpls:apply_model', 'There is no variance left in one/some of the new data observations')
                end

                for b = 1:self.B
                    % Block score
                    state.T{b}(:,a) = self.regress_func(new{b}.data, self.W{b}(:,a), new{b}.has_missing);
                    state.T{b}(:,a) = state.T{b}(:,a) .* self.block_scaling(b);
                    % Transfer it to the superscore matrix
                    state.T_sb(:,b,a) = state.T{b}(:,a);
                end

                % Calculate the superscore, T_super
                % TODO(KGD): handle missing values in T_sb (a block may not necessarily be present yet)
                state.T_super(:,a) = state.T_sb(:,:,a) * self.super.W(:,a);
                
                % Deflate each block: using the SUPERSCORE and the block loading
                for b = 1:self.B
                    deflate = state.T_super(:,a) * self.P{b}(:,a)';
                    state.stats.R2{b}(:,1) = self.ssq(deflate, 2) ./ state.stats.initial_ssq{b};
                    new{b}.data = new{b}.data - deflate;
                end
            end % looping on ``a`` latent variables
            state.Y_pred = state.T_super * self.super.C(:,1:a)';
            state.Y_pred = state.Y_pred ./ self.YPP.scaling + self.YPP.mean_center;
            
            % Summary statistics for each block and the super level
            overall_variance = zeros(state.Nnew, 1);
            for b = 1:self.B
                block_variance = self.ssq(new{b}.data, 2);
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
                    fprintf_args(2+b) = self.stats{b}.R2Xb_a(a)*100;
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
            
        end % ``summary``
        
        % Superclass abstract method implementation
        function out = register_plots_post(self)
            out = [];
            
            % Dimension 1 (rows) plots
            % =========================
            plt.name = 'Predictions';
            plt.weight = 50;
            plt.dim = 1;
            plt.more_text = 'of variable';
            plt.more_type = '<label>';
            plt.more_block = 'Y';
            plt.callback = @self.predictions_plot;
            plt.annotate = @self.predictions_annotate;
            out = [out; plt];
            
            plt.name = 'Observations';
            plt.weight = 50;
            plt.dim = 1;
            plt.more_text = 'of variable';
            plt.more_type = '<label>';
            plt.more_block = 'Y';
            plt.callback = @self.observed_plot;
            plt.annotate = @self.observed_plot_annotate;
            out = [out; plt];
            
            % Dimension 2 (columns) plots
            % ============================          
            plt.name = 'Coeffients';
            plt.weight = 30;
            plt.dim = 2;
            plt.more_text = 'for variable';
            plt.more_type = '<label>';
            plt.more_block = 'Y';
            plt.callback = @self.coefficient_plot;
            plt.annotate = @self.coefficient_plot_annotate;
            out = [out; plt];
            
            plt.name = 'Weights';
            plt.weight = 30;
            plt.dim = 2;
            plt.more_text = ': of component';
            plt.more_type = 'a';
            plt.more_block = '';
            plt.callback = @self.weights_plot;
            plt.annotate = @self.weights_plot_annotate;
            out = [out; plt];
            
            plt.name = 'R2-Y-variable';
            plt.weight = 30;
            plt.dim = 2;
            plt.more_text = ': of component';
            plt.more_type = 'a';
            plt.more_block = '';
            plt.callback = @self.R2_per_Y_variable_plot;
            plt.annotate = @self.R2_per_Y_variable_plot_annotate;
            out = [out; plt];
            
        end % ``register_plots_post``
        
        function out = get_predictions(self, varargin)
            % Gets the prediction matrix of the data in the model building
            % using ``varargin{1}`` components.
            if nargin>1
                using_A = varargin{1};
            else
                using_A = self.A;
            end
            
            if using_A == self.A            
                out = self.Y_hat.un_preprocess([], self.YPP);
            end
            
        end % ``get_predictions``
        
        function out = single_block_PLS(self, X, Y)
            % Extracts a PLS component on a single block of data, ``X`` and
            % ``Y``.
            %
            % [1] Höskuldsson, PLS regression methods, Journal of Chemometrics,
            %     2(3), 211-228, 1998, http://dx.doi.org/10.1002/cem.1180020306
            %
            % [2] Missing data: http://dx.doi.org/10.1016/B978-044452701-1.00125-3
            %
            % Returns a ``trans`` (transient) data structure that is meant to
            % be used to extract the relevant results.
            
            out = struct;
            N = size(X, 1);            
            rand('state', 0) %#ok<RAND>
            u_a_guess = rand(N,1)*2-1;
            out.u_a = u_a_guess + 1.0;
            out.itern = 0;
            while not(self.iter_terminate(u_a_guess, out.u_a, out.itern, self.opt.tolerance))
                % 0: Richardson's acceleration, or any numerical acceleration
                %    method for PLS where there is slow convergence?
                
                % Progress for PLS converges logarithmically from whatever
                % starting tolerance to the final tolerance.  Use a linear
                % mapping between 0 and 1, where 1 is mapped to log(tol).
                if out.itern == 3
                    start_perc = log(norm(u_a_guess - out.u_a));
                    final_perc = log(self.opt.tolerance);
                    progress_slope = (1-0)/(final_perc-start_perc);
                    progress_intercept = 0 - start_perc*progress_slope;
                end
                
                if self.opt.show_progress && out.itern > 2 
                    perc = log(norm(u_a_guess - out.u_a))*progress_slope + progress_intercept;
                    self.progressbar(perc);
                    
                    % Ideally the progress bar should return a "stop early"
                    % signal, allowing us to break out of the code.
                    
                    %if stop_early
                    %    break;
                    %end
                end
                
                % 0: starting point for convergence checking on next loop
                u_a_guess = out.u_a;
                
                % 1: Regress the score, u_a, onto every column in X, compute the
                %    regression coefficient and store in w_a
                % w_a = X.T * u_a / (u_a.T * u_a)
                out.w_a = self.regress_func(X, out.u_a, self.has_missing);
                
                % 2: Normalize w_a to unit length
                out.w_a = out.w_a / norm(out.w_a);
                
                % TODO(KGD): later on: investage NOT deflating X
                
                % 3: Now regress each row in X on the w_a vector, and store the
                %    regression coefficient in t_a
                % t_a = X * w_a / (w_a.T * w_a)
                out.t_a = self.regress_func(X, out.w_a, self.has_missing);

                % 4: Now regress score, t_a, onto every column in Y, compute the
                %    regression coefficient and store in c_a
                % c_a = Y.T * t_a / (t_a.T * t_a)
                out.c_a = self.regress_func(Y, out.t_a, self.has_missing);
                
                % 5: Now regress each row in Y on the c_a vector, and store the
                %    regression coefficient in u_a
                % u_a = Y * c_a / (c_a.T * c_a)
                %
                % TODO(KGD):  % Still handle case when entire row in Y is missing
                out.u_a = self.regress_func(Y, out.c_a, self.has_missing);
                
                out.itern = out.itern + 1;
            end
            
            % 6: To deflate the X-matrix we need to calculate the
            % loadings for the X-space.  Regress columns of t_a onto each
            % column in X and calculate loadings, p_a.  Use this p_a to
            % deflate afterwards.
            % Note the similarity with step 4! and that similarity helps
            % understand the deflation process.
            out.p_a = self.regress_func(X, out.t_a, self.has_missing); 
            
            if self.opt.show_progress 
                self.progressbar(1.0)
            end
            
        end % ``single_block_PLS``
        
    end % end methods (ordinary)
    
    % These methods don't require a class instance
    methods(Static=true)
        
        
        function observed_plot(hP, series)
            % Score plots for overall or block scores
            
            ax = hP.gca();
            if strcmpi(series.current, 'x')
                idx = series.x_num;
            elseif strcmpi(series.current, 'y')
                idx = series.y_num;
            end       
            
            plotdata = hP.model.Y_copy.data(:, idx);            
            
            if strcmpi(series.current, 'x')
                hPlot = hP.set_data(ax, plotdata, []);
            elseif strcmpi(series.current, 'y')
                hPlot = hP.set_data(ax, [], plotdata);
            end
            
            % If an ordered plot, connect the points, else just show dots
            if strcmpi(series.x_type{1}, 'Order')
                set(hPlot, 'LineStyle', '-', 'Marker', '.', 'Color', [0, 0, 0])
            else
                set(hPlot, 'LineStyle', 'none', 'Marker', '.', 'Color', [0, 0, 0])
            end
        end
        function observed_plot_annotate(hP, series)
            ax = hP.gca();
            if strcmpi(series.current, 'x')
                idx = series.x_num;
            elseif strcmpi(series.current, 'y')
                idx = series.y_num;
            end
            
            labels = hP.model.Y.labels{2};
            if isempty(labels)
                if strcmpi(series.current, 'x')
                    tag_name = ['tag ', num2str(series.x_num)];
                elseif strcmpi(series.current, 'y')
                    tag_name = ['tag ', num2str(series.y_num)];
                end
            else
                tag_name = labels{idx};
            end
                
            label_str = ['Observed: ', tag_name];            
            if strcmpi(series.current, 'x')
                xlabel(ax, label_str)
            elseif strcmpi(series.current, 'y')
                ylabel(ax, label_str)
            end            
            x_ax = series.x_type{1};
            y_ax = series.y_type{1};
            if strcmpi(x_ax, 'Observations') && strcmpi(y_ax, 'Predictions')
                
                title(ax, ['Obs vs predicted: ', tag_name])
                annotate_obs_predicted(hP.gca)
            end
            
            if series.x_num < 0 && series.y_num > 0
                title('Observations of: ', labels{idx})
            end
            grid on
        end
        
        function predictions_plot(hP, series)
            ax = hP.gca();
            hP.hDropdown.setEnabled(false)
            if strcmpi(series.current, 'x')
                idx = series.x_num;
            elseif strcmpi(series.current, 'y')
                idx = series.y_num;
            end   
            % Get the column labels for the Y-block
            label_str = [];
            labels = hP.model.Y.labels{2};
            if not(isempty(labels))
                label_str = ['Predicted: ', labels{idx}];
            end
            
            if strcmpi(series.current, 'x')
                xlabel(ax, label_str)
            elseif strcmpi(series.current, 'y')
                ylabel(ax, label_str)
            end
                        
            plotdata = hP.model.get_predictions();
            plotdata = plotdata(:, idx);
            
            if strcmpi(series.current, 'x')
                hPlot = hP.set_data(ax, plotdata, []);
            elseif strcmpi(series.current, 'y')
                hPlot = hP.set_data(ax, [], plotdata);
            end
            if strcmpi(series.x_type{1}, 'Order')
                set(hPlot, 'LineStyle', '-', 'Marker', '.', 'Color', [0, 0, 0], 'MarkerSize', 15)
            else
                set(hPlot, 'LineStyle', 'none', 'Marker', '.', 'Color', [0, 0, 0], 'MarkerSize', 15)
            end
        end
        function predictions_annotate(hP, series)
            ax = hP.gca();
            if strcmpi(series.current, 'x')
                idx = series.x_num;
            elseif strcmpi(series.current, 'y')
                idx = series.y_num;
            end   
            labels = hP.model.Y.labels{2};
            if isempty(labels)
                if strcmpi(series.current, 'x')
                    tag_name = ['tag ', num2str(series.x_num)];
                elseif strcmpi(series.current, 'y')
                    tag_name = ['tag ', num2str(series.y_num)];
                end
            else
                tag_name = labels{idx};
            end
                
            label_str = ['Predicted: ', tag_name];
            if strcmpi(series.current, 'x')
                xlabel(ax, label_str)
            elseif strcmpi(series.current, 'y')
                ylabel(ax, label_str)
            end
                        
            x_ax = series.x_type{1};
            y_ax = series.y_type{1};
            if strcmpi(x_ax, 'Observations') && strcmpi(y_ax, 'Predictions')                
                title(ax, ['Obs vs predicted: ', tag_name])
                annotate_obs_predicted(hP.gca)
            end
            if series.x_num < 0 && series.y_num > 0
                title('Predictions of: ', labels{idx})
            end
        end
        
        function [weights_h, weights_v, batchblock] = get_weights_data(hP, series)
            % Correctly fetches the loadings data for the current block and dropdowns  \
            % Ugly hack to get batch blocks shown            
            block = hP.c_block;
            batchblock = [];
            weights_h.data = [];
            
            % Single block data sets (PCA and PLS)
            if hP.model.B == 1
                if series.x_num > 0
                    weights_h = hP.model.W{1}(:, series.x_num);
                    if isa(hP.model.blocks{1}, 'block_batch')
                        batchblock = hP.model.blocks{1};
                    end
                end
                weights_v = hP.model.W{1}(:, series.y_num);
                if isa(hP.model.blocks{1}, 'block_batch')
                    batchblock = hP.model.blocks{1};
                end
                return
            end
            
            % Multiblock data sets: we also have superloadings
            if block == 0
                weights_v = hP.model.super.W(:, series.y_num);
            else
                weights_v = hP.model.W{block}(:, series.y_num);
                if isa(hP.model.blocks{block}, 'block_batch')
                    batchblock = hP.model.blocks{block};
                end
            end
                
            if series.x_num > 0
                if block == 0
                    weights_h = hP.model.super.W(:, series.x_num); 
                else
                    weights_h = hP.model.W{block}(:, series.x_num);
                    if isa(hP.model.blocks{block}, 'block_batch')
                        batchblock = hP.model.blocks{block};
                    end
                end
            end
        end        
        function weights_plot(hP, series)            
            % Loadings plots for overall or per-block
            ax = hP.gca();
            [weights_h, weights_v, batchblock] = hP.model.get_weights_data(hP, series);
            % Batch plots are shown differently
            % TODO(KGD): figure a better way to deal with batch blocks
            if ~isempty(batchblock) && not(strcmpi(hP.ptype, 'weights-batch'))
                hP.model.plot(hP, 'weights-batch')
                return
            end
            if strcmpi(hP.ptype, 'weights-batch')
                % Find the batchblock, set it to the current block
                hP.hDropdown.setEnabled(false)
                for b=1:hP.model.B
                    if isa(hP.model.blocks{b}, 'block_batch')
                        bblock = b;
                        hP.c_block=bblock;
                        continue
                    end
                end
                [weights_h, weights_v, batchblock] = hP.model.get_weights_data(hP, series);
            end
            % Bar plot of the single loading
            if series.x_num <= 0
                % We need a bar plot for the loadings
                hPlot = findobj(ax, 'Tag', 'lvmplot_series');
                if hPlot
                    if ishandle(hPlot)
                        delete(hPlot)
                    end
                end
                hPlot = bar(ax, weights_v, 'FaceColor', hP.opt.bar.facecolor);
                set(hPlot, 'Tag', 'lvmplot_series');
                
                % Batch plots are shown differently
                % TODO(KGD): figure a better way to deal with batch blocks
                if ~isempty(batchblock)
                    hP.annotate_batch_trajectory_plots(ax, hPlot, batchblock)
                end
                
                return
            end
            
            % Scatter plot of the loadings            
            hPlot = hP.set_data(ax, weights_h, weights_v);            
            set(hPlot, 'LineStyle', 'none', 'Marker', '.', 'Color', [0, 0, 0]);
            set(ax, 'XLim', hP.get_good_limits(weights_h, get(ax, 'YLim'), 'zero'))
            set(ax, 'YLim', hP.get_good_limits(weights_v, get(ax, 'YLim'), 'zero'))
        end
        function weights_plot_annotate(hP, series)
            ax = hP.gca();
            hBar = findobj(ax, 'Tag', 'lvmplot_series');
            if isempty(hBar)
                return
            end
            if series.x_num > 0 && series.y_num > 0                         
                title(ax, 'Weights plot')
                grid on
                xlabel(ax, ['w_', num2str(series.x_num)])
                ylabel(ax, ['w_', num2str(series.y_num)])
                hPlot = findobj(ax, 'Tag', 'lvmplot_series');
                if hP.model.B > 1
                    labels = hP.model.get_labels(hP.dim, hP.c_block);
                else
                    labels = hP.model.get_labels(hP.dim, 1);
                end
                hP.label_scatterplot(hPlot, labels);
                
            
            elseif series.y_num > 0                
                title(ax, 'Weights bar plot', 'FontSize', 15);
                grid on
                ylabel(ax, ['w_', num2str(series.y_num)])
                
                hBar = findobj(ax, 'Tag', 'lvmplot_series');
                if hP.model.B > 1
                    labels = hP.model.get_labels(hP.dim, hP.c_block);
                else
                    labels = hP.model.get_labels(hP.dim, 1);
                end
                hP.annotate_barplot(hBar, labels)
            end
        end
        
        function R2_per_Y_variable_plot(hP, series)
            ax = hP.gca();
            hP.hDropdown.setEnabled(false)
            R2_data = hP.model.super.stats.R2Yk_a(:, 1:series.y_num);
            if series.x_num <= 0
                
                hPlot = findobj(ax, 'Tag', 'lvmplot_series');
                if hPlot
                    if ishandle(hPlot)
                        delete(hPlot)
                    end
                end
                hPlot = bar(ax, R2_data, 'stacked', 'FaceColor', [1,1,1]);
                set(hPlot, 'Tag', 'lvmplot_series');
                set(ax, 'YLim', [0.0, 1.0])
            end
        end
        function R2_per_Y_variable_plot_annotate(hP, series)
            ax = hP.gca();
            hBar = findobj(ax, 'Tag', 'lvmplot_series');
            
            if series.x_num > 0 && series.y_num > 0                         
                title(ax, 'R2 plot for Y-variables')
                grid on
                xlabel(ax, ['R2 with A=', num2str(series.x_num)])
                ylabel(ax, ['R2 with A=', num2str(series.y_num)])
                
            elseif series.y_num > 0                
                title(ax, 'R2 bar plot for Y-variables')
                grid on
                ylabel(ax, ['R2 for Y-variables with A=', num2str(series.y_num)])
                
                labels = hP.model.Y.labels;
                if isempty(labels)
                    tagnames = [];
                else
                    tagnames = labels{2};
                end                
                hP.annotate_barplot(hBar, tagnames, 'stacked')
            end
        end
        
        
    end % end methods (static)
    
end % end classdef

function annotate_obs_predicted(ax)    
    
    if findobj(ax, 'Tag', 'RMSEE_obs_pred')
        return
    end
    extent = axis;
    min_ex = min(extent([1,3]));
    max_ex = min(extent([2,4]));
    delta = (max_ex - min_ex);
    min_ex_l = min_ex - delta*1.5;
    max_ex_l = max_ex + delta*1.5;
    set(ax, 'NextPlot', 'add')
    hd = plot(ax, [min_ex_l, max_ex_l], [min_ex_l, max_ex_l], 'k', 'linewidth', 2);
    set(hd, 'tag', 'hline', 'HandleVisibility', 'on')    
    hData = findobj(ax, 'Tag', 'lvmplot_series');
    x_data = get(hData, 'XData');
    x_data(isnan(x_data)) = [];
    y_data = get(hData, 'YData');
    y_data(isnan(y_data)) = [];    
    RMSEE = sqrt(mean((x_data - y_data).^2));
    hText = text(min_ex + 0.05*delta, max_ex - 0.05*delta, sprintf('RMSEE = %0.4g', RMSEE));
    set(hText, 'Tag', 'RMSEE_obs_pred');

    xlim([min_ex-0.1*delta, max_ex+0.1*delta])
    ylim([min_ex-0.1*delta, max_ex+0.1*delta])
end
