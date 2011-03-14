% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for defining, building and manipulating latent variable models.


% Create a handle class
% This is the highest level latent variable model class
classdef mblvm < handle
    properties
        model_type = '';    % for convenience, shouldn't be used in code; used to display model type
        blocks = {};        % a cell array of data blocks; last block is always "Y" (can be empty)
        A = 0;              % number of latent variables
        B = 0;              % number of blocks
        K = 0;              % size (width) of each block in the model
        N = 0;              % number of observations in the model
        M = 0;              % number of variables in the Y-block (if present)
        
        opt = struct();     % model options
        stats = cell({});   % Model statistics for each block
        model = cell({});   % Model-related propert (timing, iterations, risk)
        lim = cell({});     % Model limits
        
        % Model parameters for each block (each cell entry is a block)
        P = cell({});       % Block loadings, P
        T = cell({});       % Block scores, T        
        W = cell({});       % Block weights (for PLS only)
        R = cell({});       % Block weights (W-star matrix, for PLS only) 
        C = cell({});       % Block loadings (for PLS only)
        U = cell({});       % Block scores (for PLS only) 
        S = cell({});       % Variance-covariance matrix for the scores
        beta = cell({});    % Beta-regression coeffients 
        super = struct();   % Superblock parameters (weights, scores)
        PP = cell({});      % Preprocess model parameters 
        
        % Removed: this is not orthogonal: best we calculate it when required
        %S = cell({});       % Std deviation of the scores
        
        % Specific to batch models
        T_j = [];    % Instantaneous scores
        error_j = [];% Instantaneous errors
    end 
    
    % Only set by this class and subclasses
    properties (SetAccess = protected)
        data = [];
        has_missing = false;
        block_scaling = [];
    end
    
    % Subclasses may choose to redefine these methods
    methods
        function self = mblvm(data, varargin)
            
            self.blocks = data;
            
            % Process model options, if provided
            % -----------------------
            self.opt = lvm_opt('default');

            % Build the model if ``A``, is specified as a
            % numeric value.  If it is a structure of 
            % options, then override any default settings with those. 
            
            if nargin == 1
                varargin{1} = 0;
            end

            % Merge options: only up to 2 levels deep
            if isa(varargin{1}, 'struct')
                options = varargin{1};
                names = fieldnames(options);
                for k = 1:numel(names)
                    if ~isa(options.(names{k}), 'struct')
                        self.opt.(names{k}) = options.(names{k});
                    else
                        names2 = fieldnames(options.(names{k}));
                        for j = 1:numel(names2)
                            if ~isa(options.(names{k}).(names2{j}), 'struct')
                                self.opt.(names{k}).(names2{j}) = options.(names{k}).(names2{j});
                            else
                            end
                        end
                    end
                end

            % User is specifying "A"
            elseif isa(varargin{1}, 'numeric')
                self.opt.build_now = true; 
                self.opt.min_lv = varargin{1};
                self.opt.max_lv = varargin{1};
            end
            
            % Create storage structures
            self.create_storage();
             
        end % ``lvm``
        
        function out = get.B(self)
            % Purely a convenience function to get ".B"
            out = numel(self.blocks);
        end
        
        function N = get.N(self)
            N = size(self.blocks{1}.data, 1);
            for b = 1:self.B
                assert(N == self.blocks{b}.N);
            end
        end
        
        function K = get.K(self)
            K = zeros(1, self.B);            
            for b = 1:self.B
                K(b) = self.blocks{b}.K;
            end
        end
        
        function idx = b_iter(self, varargin)
            % An iterator for working across a series of merged blocks
            % e.g. if we have 10, 15 and 18 columns, giving a total of 43 cols
            %      idx = [[ 1, 10],
            %             [11, 25],
            %             [26, 43]]
            % Quick result: e.g. if self.b_iter(2), then it will return a 
            %               the indices 11:25.
            if nargin > 1
                splits = self.K;
                start = 1;                
                for b = 2:varargin{1}
                    start = start + splits(b-1);                    
                end
                finish = start + splits(varargin{1}) - 1;
                idx = start:finish;
                return
            end
            idx = zeros(self.B, 2) .* NaN;
            idx(1,1) = 1;
            for b = 1:self.B
                if b>1
                    idx(b,1) = idx(b-1,2) + 1;
                end
                idx(b,2) = idx(b,1) + self.K(b) - 1;
            end
        end
        
        function out = iter_terminate(self, lv_prev, lv_current, itern, tolerance, conditions)
            % The latent variable algorithm is terminated when any one of these
            % conditions is True
            %  #. scores converge: the norm between two successive iterations
            %  #. a max number of iterations is reached
            score_tol = norm(lv_prev - lv_current);                       
            conditions.converged = score_tol < tolerance;
            conditions.max_iter = itern > self.opt.max_iter;            
            
            if conditions.converged || conditions.max_iter
                out = true;  % algorithm has converged
            else
                out = false;
            end
        end % ``iter_terminate``
        
        function disp(self)
            % Displays a text summary of the model
            if self.A == 0
                word = ' components (unfitted)';
            elseif self.A == 1
                word = ' component';
            else
                word = ' components';
            end
            disp(['Latent Variable Model: ', upper(self.model_type), ' model: '...
                   num2str(self.A), word, '.'])
               
            if self.A == 0
                return
            end
            
            % Call the subclass to provide more information
            self.summary()
        end % ``disp``
        
        function self = merge_blocks(self)
            % Merges all X-blocks together after preprocessing, but applies
            % the overall block scaling
            
            self.block_scaling = 1 ./ sqrt(self.K);
            if self.B == 1
                self.block_scaling = 1;
            end
            
            if isempty(self.data)
                self.data = ones(self.N, sum(self.K)) .* NaN;
                self.has_missing = false;            
                for b = 1:self.B
                    self.data(:, self.b_iter(b)) = self.blocks{b}.data .* self.block_scaling(b);
                    if self.blocks{b}.has_missing
                        self.has_missing = true;
                    end
                end
            end
        end % ``merge_blocks``  
        
        function self = calc_statistics_and_limits(self, a)
            % Calculate summary statistics and limits for the model. 
            % TODO
            % ----
            % * Modelling power of each variable
            % * Eigenvalues
            
            % Check on maximum number of iterations
            if any(self.model.stats.itern >= self.opt.max_iter)
                warn_string = ['The maximum number of iterations was reached ' ...
                    'when calculating the latent variable(s). Please ' ...
                    'check the raw data - is it correct? and also ' ...
                    'adjust the number of iterations in the options.'];
                warning('mbpca:calc_statistics_and_limits', warn_string)
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
            %         degrees of freedom.  Based on the central limit theorem.
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
            
            % Super-level statistics. NOTE: the superlevel statistics will
            % agree with the block level statistics and limits if there is
            % only a single block.
            self.super.lim.SPE(a) = self.spe_limits(self.super.SPE(:,a), siglevel, sum(self.K));
            self.super.lim.T2(a) = self.T2_limits(self.super.T(:,1:a), siglevel, a);
            self.super.lim.t(a) = self.score_limits(self.super.T(:, a), siglevel);
            
            for b = 1:self.B
                self.lim{b}.SPE(a) = self.spe_limits(self.stats{b}.SPE(:,a), siglevel, self.K(b));
                self.lim{b}.T2(a) = self.T2_limits(self.T{b}(:, 1:a), siglevel, a);
                self.lim{b}.t(a) = self.score_limits(self.T{b}(:, a), siglevel);
            end 
            
        end % ``calc_statistics_and_limits``
        
    end % end: methods (ordinary)
    
    % Subclasses may not redefine these methods
    methods (Sealed=true)
                
        function self = build(self, varargin)
            % Build the multiblock latent variable model.
            % * preprocess the data if it has not been already
          
            if nargin==2 && isa(varargin{1}, 'numeric')
                given_A = varargin{1};
            else
                given_A = 0;
            end

            requested_A = max([self.opt.min_lv, 0, given_A]);
            
            % TODO: handle the case where the model is shrunk or grown to 
            %       a different A value.
            % Resize the storage for ``A`` components
            self.initialize_storage(requested_A);
                
            preprocess_blocks(self);        % superclass method            
            merge_blocks(self);             % method may be subclassed 
            calc_model(self, requested_A);  % method must be subclassed
            limit_calculations(self);       % method must be subclassed
        end % ``build``
        
        function limit_calculations(self)
            % Calculates the monitoring limits for a batch blocks in the model
            
            batch_blocks = false(1, self.B);
            for b = 1:self.B
                if any(ismember(properties(self.blocks{b}), 'batch_raw'))
                    batch_blocks(b) = true;
                end
            end
            
            batch_blocks = find(batch_blocks);
            if isempty(batch_blocks)
                return
            end
            if numel(batch_blocks) > 1
                error('Currently only supports a single batch block.')
            end                
            
            batchstat = cell(self.B, 1);
            for b = 1:batch_blocks
                batchstat{b} = struct;
                self.stats{b}.T_j = zeros(self.N, self.A, self.blocks{b}.J);
                %batchstat{b}.SPE_j_temp = zeros(self.N, self.blocks{b}.J);
                batchstat{b}.error_j = zeros(self.N, self.K);
                J_time = self.blocks{b}.J;
            end
            superstat = struct;
            superstat.SPE = zeros(self.N, J_time);
            superstat.T2 = zeros(self.N, J_time);
            
            show_progress = true; %%self.opt.show_progress;
            if show_progress
                h = awaitbar(0, sprintf('Calculating monitoring limits for model'));
            end
            stop_early = false;
            
                
            % Iterate over every observation in the data set (row)            
            apply_opt = struct;
            apply_opt.quick = true;
            apply_opt.preprocess = false;
            apply_opt.data = [];
            
            % TODO(KGD): make it more general: allow multiple batch blocks
            for n = 1:self.N                
                if show_progress
                    perc = floor(n/self.N*100);
                    stop_early = awaitbar(perc/100,h,sprintf('Processing batches for limits %d. [%d%%]',n, perc));
                    if stop_early                             
                        break; 
                    end	
                end
                
                % Assemble the raw data for all blocks
                
                raw_n = cell(1, self.B);
                for b = 1:self.B
                    
                    % Short cut name
                    blk = self.blocks{b};
                    
                    % If this is a batch block, then we also start to unfold
                    % with time
                    if any(b==batch_blocks)
                        
                        trajectory_n = blk.data(n,:);
                        
                        % Let the trajectory just contain missing values
                        raw_n{b} = ones(size(trajectory_n)) .* NaN;
                        for j = 1:blk.J
                            idx_beg = blk.nTags*(j-1)+1;
                            idx_end = blk.nTags*j;

                            % Store the preprocessed vector back in the array
                            raw_n{b}(1, idx_beg:idx_end) = trajectory_n(1, idx_beg:idx_end);
                            
                            out = self.apply(raw_n, apply_opt);
                            superstat.SPE(n,j) = out.stats.super.SPE;
                            superstat.T2(n,j) = out.stats.super.T2;
                            self.stats{b}.T_j(n, :, j) = out.T{b};
                            batch.error_j(n, idx_beg:idx_end) = out.newb{b}.data(1, idx_beg:idx_end);
                            %SPE_j_temp(n,j) = ssq(out.data(1, idx_beg:idx_end));
                            %batch.stats.SPE_j(n, j) = out.stats.SPE(1, end); % use only the last component
                             
                        end % ``j=1, 2, ... J``
                    else
                        raw_n{b} = blk.data(n, :);
                    end % ``if-else: a batch block ?
                    
                end % ``b = 1, 2, ... self.B``
                
                % Apply the model to these new data
                
                
               
                
                %out = self.apply_PCA(batch.lim.x_pp, self.A, 'quick');
                %out = self.apply_PLS(batch.lim.x_pp, [], self.A, 'quick');
                %    batch.data_pred_pp = out.T * batch.C';
                %    batch.data_pred = batch.data_pred_pp / batch.PP.scaling + batch.PP.mean_center;
                
                
            end % ``n=1, 2, ... N`` 

            
            % Next, calculate time-varying limits for each block
            % ---------------------------------------------------
            if not(stop_early)
                siglevel = 0.95;
                
                % Similar code exists in ``calc_statistics_and_limits``
                
                for b = 1:self.B
                    if any(b==batch_blocks)
                        
                        % J by A matrix
                        std_t_j = squeeze(self.score_limits(self.stats{b}.T_j, siglevel))';
                        % From the central limit theorem, assuming each batch is 
                        % independent from the next (because we are calculating
                        % out std_t_j over the batch dimension, not the time
                        % dimension).  Clearly each time step is not independent.
                        % We inflate the limits slightly: see Nomikos and
                        % MacGregor's article in Technometrics
                        self.lim{b}.t_j = std_t_j .* (1 + 1/self.blocks{b}.N);
                        

                        % Variance/covariance of the score matrix over time.
                        % The scores, especially at the start of the batch are
                        % actually correlated; only at the very end of the
                        % batch do they become uncorrelated
                        for j = 1:self.blocks{b}.J
                            scores = self.stats{b}.T_j(:,:,j);
                            sigma_T = inv(scores'*scores/(self.blocks{b}.N-1));
                            for n = 1:self.blocks{b}.N
                                self.stats{b}.T2_j(n, j) = self.stats{b}.T_j(n,:,j) * sigma_T * self.stats{b}.T_j(n,:,j)';
                            end

                        end
                        % Calculate the T2 limit
                        % Strange that you can calculate it without reference to
                        % any of the batch data.  Also, I would have expected it
                        % to vary during the batch, because t1 and t2 limits are
                        % large initially.
                        
                        
                        %N_lim = self.blocks{b}.N;
                        for a = 1:self.A
                            self.lim{1}.T2_j = T2_limits(self.stats{b}.T_j, siglevel, a);
                            
                            
                            %mult = a*(N_lim-1)*(N_lim+1)/(N_lim*(N-a));
                            %limit = my_finv(alpha, a, N-(self.A));
                            %batch.lim.T2_j(:,a) = mult * limit;
                            % This value should agree with batch.lim.T2(:,a)
                            % TODO(KGD): slight discrepancy in batch SBR dataset
                        end

                        % Calculate SPE time-varying limits
                        % Our SPE definition = sqrt(e'e / K), where K = number of
                        % entries in vector ``e``.  In the SPE_j case that is the
                        % number of tags.
                        % Apply smoothing window here: see Nomikos thesis, p 66.
                        % ``w`` should be a function of how "jumpy" the SPE plot
                        % looks.  Right now, I'm going to set ``w`` proportional
                        % to the number of time samples in a batch
                        w = max(1, ceil(0.012/2 * batch.J));
                        for j = 1:batch.J
                            start_idx = max(1, j-w);
                            end_idx = min(batch.J, j+w);
                            SPE_values = SPE_j_temp(:, start_idx:end_idx);
                            batch.lim.SPE_j(j) = sqrt(calculate_SPE_limit(SPE_values, alpha)/batch.nTags);
                        end
                        SPE_j_temp = SPE_j_temp ./ batch.nTags;
                        batch.stats.SPE_j = sqrt(SPE_j_temp);
                        
                    end % ``if: a batch block ?
                end % ``b = 1, 2, ... self.B``
            end %``not stop_early``
            
            limits_subclass(self)
            
        end % ``limit_calculations``
         
        function state = apply(self, new, varargin)
            % Apply the multiblock latent variable model to new data.
            % * preprocess the data if it has not been already
            
            % Default options
            apply_opt = struct;
            apply_opt.complete = false;
            apply_opt.preprocess = true;
            apply_opt.data = [];
                
            if nargin == 3
                options = varargin{1};
                names = fieldnames(options);
                for k = 1:numel(names)
                    apply_opt.(names{k}) = options.(names{k});
                end
            end
            
            if not(isempty(apply_opt.data))
                newb = apply_opt.data;
            else                
                Nnew = 0;
                newb = cell(self.B, 1);
                for b = 1:numel(new)
                    newb{b} = block(new{b});                    
                    Nnew = max(Nnew, newb{b}.N);
                end
            end

            if apply_opt.preprocess
                for b = 1:self.B
                    newb{b} = self.blocks{b}.preprocess(newb{b}, self.PP{b});
                end
            end
            
            % Initialize the states (this could go in another function later)
            state = struct;
            state.opt = apply_opt;
            state.Nnew = Nnew;
            for b = 1:self.B
                % Block scores
                state.T{b} = ones(Nnew, self.A) .* NaN;
            end
            % Superblock collection of scores: N x B x A
            state.T_sb = ones(Nnew, self.B, self.A) .* NaN;
            
            % Super scores: N x A
            state.T_super = ones(Nnew, self.A) .* NaN;
            
            % Summary statistics
            state.stats.T2 = cell(1, self.B);
            state.stats.SPE = cell(1, self.B);
            state.stats.super.T2 = ones(Nnew, 1) .* NaN;
            state.stats.super.SPE = ones(Nnew, 1) .* NaN;                        
            state.stats.super.R2 = ones(Nnew, 1) .* NaN;
            state.stats.initial_ssq_total = ones(Nnew, 1) .* NaN;
            for b = 1:self.B
                state.stats.R2{b} = ones(Nnew, 1) .* NaN;
            end            
            state.stats.R2 = cell(1, self.B);
            state.stats.initial_ssq = cell(1, self.B);

            state = apply_model(self, newb, state); % method must be subclassed
            state.newb = newb;
        end % ``apply``
        
        function self = create_storage(self)
            % Creates only the subfields required for each block.
            % NOTE: this function does not depend on ``A``.  That storage is
            %       sized in the ``initialize_storage`` function.
            
            % General model statistics
            % -------------------------
            % Time used to calculate each component
            self.model.stats.timing = [];
            
            % Iterations per component
            self.model.stats.itern = [];             
            
            nb = self.B; 
            
            % Latent variable parameters
            self.P = cell(1,nb);
            self.T = cell(1,nb);            
            self.W = cell(1,nb);
            self.R = cell(1,nb);
            self.C = cell(1,nb);
            self.U = cell(1,nb);
            self.S = cell(1,nb);

            % Block preprocessing 
            self.PP = cell(1,nb);
            
            % Statistics and limits for each block
            self.stats = cell(1,nb);
            self.lim = cell(1,nb);
            
            % Super block parameters
            self.super = struct();
            
            % Collected scores from the block into the super level
            % N x B x A
            self.super.T_summary = [];
            % Superblock scores
            % N x A
            self.super.T = [];
            % Superblock scores variance-covariance matrix
            % A x A
            self.super.S = [];
            % Superblock's loadings (used in PCA only)
            % B x A
            self.super.P = [];    
            % Superblocks's weights (used in PLS only)
            % B x A
            self.super.W = [];
            % Superblocks's weights for Y-space (used in PLS only)
            % M x A
            self.super.C = [];
            % Superblocks's scores for Y-space (used in PLS only)
            % N x A
            self.super.U = [];
            
            % R2 for each component for the overall model
            self.super.stats.R2X = [];
            self.super.stats.R2Y = [];
            self.super.stats.R2Yk_a = [];
            self.super.stats.SSQ_exp = [];
            % The VIP for each block, and the VIP calculation factor
            self.super.stats.VIP = [];
            self.super.stats.VIP_f = cell({});
            
            % SPE, after each component, for the overall model
            self.super.SPE = [];            
            % T2, using all components 1:A, for the overall model
            self.super.T2 = [];            
           
            % Limits for various parameters in the overall model
            self.super.lim = cell({});
        end % ``create_storage``
        
        function self = initialize_storage(self, A)
            % Initializes storage matrices with the correct dimensions.
            % If ``A`` is given, and ``A`` > ``self.A`` then it will expand
            % the storage matrices to accommodate the larger number of
            % components.
            if A <= self.A
                A = self.A;
            end
            
            % Does the storage size match with the number of blocks? If not,
            % then wipe out all storage, and resize to match the number of
            % blocks.
            if numel(self.P) ~= self.B
                create_storage(self);
            end
            
            self.model.stats.timing = zeroexp([A, 1], self.model.stats.timing);
            self.model.stats.itern = zeroexp([A, 1], self.model.stats.itern);            
            
            % Storage for each block
            for b = 1:self.B
                dblock = self.blocks{b};
                self.P{b} = zeroexp([dblock.K, A], self.P{b});  % block loadings; 
                self.T{b} = zeroexp([dblock.N, A], self.T{b});  % block scores
                self.W{b} = zeroexp([dblock.K, A], self.W{b});  % PLS block weights
               %self.R{b} = zeroexp([dblock.K, A], self.R{b});  % PLS block weights
                self.S{b} = zeroexp([A, A], self.S{b});         % score scaling factors
                
                % Block preprocessing options: resets them
                if numel(self.PP{b}) == 0
                    self.PP{b} = struct('mean_center', [], ...
                                        'scaling', [], ...
                                        'is_preprocessed', false);
                end

                % Calculated statistics for this block
                if numel(self.stats{b}) == 0
                    self.stats{b}.SPE = [];
                    self.stats{b}.SPE_j = [];

                    self.stats{b}.start_SS_col = [];
                    self.stats{b}.R2Xk_a = [];  % not cumulative !                    
                    self.stats{b}.col_ssq_prior = [];
                    self.stats{b}.R2b_a = [];
                    self.stats{b}.SSQ_exp = [];
                    self.stats{b}.VIP_a = [];
                    %self.stats{b}.VIP_f = cell({});

                    self.stats{b}.T2 = [];
                    self.stats{b}.T2_j = [];

                    self.stats{b}.model_power = [];
                end
                if numel(self.lim{b}) == 0
                    % Ordinary model portion
                    self.lim{b}.t = [];
                    self.lim{b}.T2 = [];
                    self.lim{b}.SPE = [];

                    % Instantaneous (batch) model portion
                    self.lim{b}.t_j = [];                
                    self.lim{b}.SPE_j = []; 
                    self.lim{b}.T2_j = []; %not used: we monitoring based on final T2 value
                end
                if numel(self.super.lim) == 0
                    % Ordinary model portion
                    self.super.lim.t = [];
                    self.super.lim.T2 = [];
                    self.super.lim.SPE = [];
                end
                
                % SPE per block
                % N x A
                self.stats{b}.SPE = zeroexp([dblock.N, A], self.stats{b}.SPE);
                
                
                % Instantaneous SPE limit using all A components (batch models)
                % N x J
                %self.stats{b}.SPE_j = zeroexp([dblock.N, dblock.J], self.stats{b}.SPE_j, true);

                
                % Baseline value for all R2 calculations: before any components are
                % extracted, but after the data have been preprocessed.
                % 1 x K(b)
                self.stats{b}.start_SS_col = zeroexp([1, dblock.K], self.stats{b}.start_SS_col);
                
                % R^2 for every variable in the block, per component (not cumulative)
                % K(b) x A
                self.stats{b}.R2Xk_a = zeroexp([dblock.K, A], self.stats{b}.R2Xk_a);
                
                % R^2 for every variable in the Y-block, per component (not cumulative)
                % M x A
                self.super.stats.R2Yk_a = zeroexp([self.M, A], self.super.stats.R2Yk_a);
                
                % Sum of squares for each column in the block, prior to the component
                % being extracted.
                % K(b) x A
                self.stats{b}.col_ssq_prior = zeroexp([dblock.K, A], self.stats{b}.col_ssq_prior);                
                                
                % R^2 for the block, per component
                % 1 x A
                self.stats{b}.R2b_a = zeroexp([1, A], self.stats{b}.R2b_a);
                
                % Sum of squares explained for this component
                % 1 x A
                self.stats{b}.SSQ_exp = zeroexp([1, A], self.stats{b}.SSQ_exp);

                % VIP value using all 1:A components (only last column is useful though)
                % K(b) x A
                self.stats{b}.VIP_a = zeroexp([dblock.K, A], self.stats{b}.VIP_a);
                
                % Overall T2 value for each observation in the block using
                % all components 1:A
                % N x A
                self.stats{b}.T2 = zeroexp([dblock.N, A], self.stats{b}.T2);
                
                % Instantaneous T2 limit using all A components (batch models)
                %self.stats{b}.T2_j = zeroexp([dblock.N, dblock.J], self.stats{b}.T2_j);

                % Modelling power = 1 - (RSD_k)/(RSD_0k)
                % RSD_k = residual standard deviation of variable k after A PC's
                % RSD_0k = same, but before any latent variables are extracted
                % RSD_0k = 1.0 if the data have been autoscaled.
                %self.stats{b}.model_power = zeroexp([1, dblock.K], self.stats{b}.model_power);

                % Actual limits for the block: to be calculated later on
                % ---------------------------
                % Limits for the (possibly time-varying) scores
                % 1 x A
                % NOTE: these limits really only make sense for uncorrelated
                % scores (I don't think it's too helpful to monitor based on limits 
                % from correlated variables)
                self.lim{b}.t = zeroexp([1, A], self.lim{b}.t);
                %self.lim{b}.t_j = zeroexp([dblock.J, A], self.lim{b}.t_j, true); 

                % Hotelling's T2 limits using A components (column)
                % (this is actually the instantaneous T2 limit,
                % but we don't call it that, because at time=J the T2 limit is the
                % same as the overall T2 limit - not so for SPE!).
                % 1 x A
                self.lim{b}.T2 = zeroexp([1, A], self.lim{b}.T2);            

                % Overall SPE limit for the block using 1:A components (use the 
                % last column for the limit with all A components)
                % 1 x A
                self.lim{b}.SPE = zeroexp([1, A], self.lim{b}.SPE);

                % SPE instantaneous limits using all A components
                %self.lim{b}.SPE_j = zeroexp([dblock.J, 1], self.lim{b}.SPE_j);

            end
            
            % Superblock storage
            % ------------------
            
            % Summary scores from each block: these match self.T{b}, so we
            % don't really need to store them in the superblock structure.
            self.super.T_summary = zeroexp([self.N, self.B, A], self.super.T_summary);
            self.super.T = zeroexp([self.N, A], self.super.T);
            self.super.P = zeroexp([self.B, A], self.super.P);
            self.super.W = zeroexp([self.B, A], self.super.W);
            self.super.C = zeroexp([self.M, A], self.super.C);  % PLS Y-space loadings
            self.super.U = zeroexp([self.N, A], self.super.U);  % PLS Y-space scores
               
            
            % T2, using all components 1:A, in the superblock
            self.super.T2 = zeroexp([self.N, A], self.super.T2);
            
            % Limits for superscore entities
            self.super.lim.t = zeroexp([1, A], self.super.lim.t);
            self.super.lim.T2 = zeroexp([1, A], self.super.lim.T2);            
            self.super.lim.SPE = zeroexp([1, A], self.super.lim.SPE);
            
            % Statistics in the superblock
            self.super.stats.R2X = zeroexp([1, A], self.super.stats.R2X);
            self.super.stats.R2Y = zeroexp([1, A], self.super.stats.R2Y);
            self.super.stats.ssq_Y_before = [];
            
            self.super.stats.SSQ_exp = zeroexp([1, A], self.super.stats.SSQ_exp);
            % The VIP's for each block, and the VIP calculation factor
            self.super.stats.VIP = zeroexp([self.B, 1], self.super.stats.VIP);
            %self.super.stats.VIP_f = cell({});
            
            % Give the subclass the chance to expand storage, if required
            self.expand_storage(A);
        end % ``initialize_storage``
        
        function self = preprocess_blocks(self)
            % TODO(KGD): preprocessing should be moved to its own class
            %            where it can handle "self" preprocessing and on 
            %            "other" blocks; and more general preprocessing options.
            % Preprocesses each block.            
            for b = 1:self.B
                if ~self.PP{b}.is_preprocessed
                    [self.blocks{b}, PP_block] = self.blocks{b}.preprocess();
                    self.PP{b}.is_preprocessed = true;
                    self.PP{b}.mean_center = PP_block.mean_center;
                    self.PP{b}.scaling = PP_block.scaling;                    
                end                
            end
            self.preprocess_extra();         % method must be subclassed 
        end % ``preprocess_blocks``
                
        function self = split_result(self, result, rootfield, subfield)
            % Splits the result from a merged calculation (over all blocks)
            % into the storage elements for each block.  It uses block size,
            % self.blocks{b}.K to learn where to split the result.
            %
            % e.g. self.split_result(loadings, 'P', '')
            %
            % Before:
            %     loadings = [0.2, 0.3, 0.4, 0.1, 0.2]
            % After:
            %     self.P{1} = [0.2, 0.3, 0.4]
            %     self.P{2} = [0.1, 0.2]
            % 
            % e.g. self.split_result(SS_col, 'stats', 'start_SS_col')
            %
            % Before: 
            %     ss_col = [6, 6, 6, 9, 9]
            %
            % After:
            %     self.stats{1}.start_SS_col = [6, 6, 6]
            %     self.stats{2}.start_SS_col = [9, 9]
            
            n = size(result, 1);
            for b = 1:self.B
                self.(rootfield){b}.(subfield)(1:n,:) = result(:, self.b_iter(b));
            end
        end

        function varargout = plot(self, varargin)
            % SYNTAX
            %
            % > plot(model)                        % summary plot of T2, SPE, R2X per PC
            % > plot(model, 'scores')              % This line and the next do the same
            % > plot(model, 'scores', 'overall')   % All scores for overall block
            % > plot(model, 'scores', {'block', 1})% All scores for block 1
            % > plot(model, {'scores', 2}, {'block', 1})  % t_2 scores for block 1
            %
            % Scores for block 1 (a batch block), only showing batch 38
            % > plot(model, 'scores', {'block', 1}, {'batch', 38)  
            
            
            % Set the defaults in the ``popt``: plot options
            popt = struct;
            popt.footer_string = {datestr(now)};
            popt.show_labels = true;
            popt.block = 'overall';
            popt.components = 1:self.A;
            popt.other = {};
            
            for i = 1:numel(varargin)
                key = varargin(i);
                value = [];
                if iscell(key)
                    if ischar(key{:})
                        key = key{1};
                    else
                        key = varargin{i}{1};
                        if numel(varargin{i}) > 1
                            value = varargin{i}{2};
                        end 
                    end
                end
                
                if strcmpi(key, 'scores')
                    show_what = 'scores';
                    if ~isempty(value)
                        popt.components = value;
                    end
                elseif strcmpi(key, 'loadings')
                    show_what = 'loadings';
                    if ~isempty(value)
                        popt.components = value;
                    end
                elseif strcmpi(key, 'block')
                    if ~isempty(value)
                        if ischar(value)
                            for b = 1:self.B
                                if strcmpi(self.blocks{b}.name, value)
                                    popt.block = b;
                                end
                            end
                        else
                            popt.block = uint32(value);
                        end
                    end
                elseif strcmpi(key, 'T2')
                    show_what = 'T2';
                elseif strcmpi(key, 'SPE')
                    show_what = 'SPE';
                elseif strcmpi(key, 'R2')
                    show_what = 'R2';
                elseif strcmpi(key, 'show_labels')
                    popt.show_labels = value;
                else
                    popt.other = {varargin(i), popt.other{:}};
                end 
            end
            
            if isinteger(popt.block) && isa(self.blocks{popt.block}, 'block_batch')
                switch show_what
                    case 'scores'
                        [h, hHeaders, hFooters, title_str] = plot_batch_scores(self, popt);
                    case 'loadings'
                        [h, hHeaders, hFooters, title_str] = plot_batch_loadings(self, popt);
                    case 'T2'
                        [h, hHeaders, hFooters, title_str] = plot_batch_T2(self, popt);
                    case 'R2'
                        [h, hHeaders, hFooters, title_str] = plot_batch_R2(self, popt);
                    case 'SPE'
                        [h, hHeaders, hFooters, title_str] = plot_batch_SPE(self, popt);
                    otherwise
                        [h, hHeaders, hFooters, title_str] = plot_batch_summary(self, popt);
                end

            end
            
            switch show_what
                case 'scores'
                    [h, hHeaders, hFooters, title_str] = plot_scores(self, popt);
                case 'loadings'
                    [h, hHeaders, hFooters, title_str] = plot_loadings(self, popt);
                case 'T2'
                    [h, hHeaders, hFooters, title_str] = plot_T2(self, popt);
                case 'R2'
                    [h, hHeaders, hFooters, title_str] = plot_R2(self, popt);
                case 'SPE'
                    [h, hHeaders, hFooters, title_str] = plot_SPE(self, popt);
                otherwise 
                    [h, hHeaders, hFooters, title_str] = plot_summary(self, popt);
            end
            
            self.add_plot_footers(hFooters, popt.footer_string);
            self.add_plot_window_title(hHeaders, title_str)
            for i=1:nargout
                varargout{i} = h;
            end
        end % ``plot``
        
    end % end: methods (sealed)
    
    % Subclasses must redefine these methods
    methods (Abstract=true)
        preprocess_extra(self)   % extra preprocessing steps that subclass does
        calc_model(self, A)      % fit the model to data in ``self``
        limits_subclass(self)    % post model calculation: fit additional limits
        apply_model(self, other) % applies the model to ``new`` data
        expand_storage(self, A)  % expands the storage to accomodate ``A`` components
        summary(self)            % show a text-based summary of ``self``
    end % end: methods (abstract)
    
    methods (Sealed=true, Static=true)
        function [T2, varargout] = mahalanobis_distance(T, varargin)
            % If given ``S``, an estimate of the variance-covariance matrix,
            % then it will use that, instead of an estimated var-cov matrix.
            
            % TODO(KGD): calculate this in a smarter way. Can create unnecessarily
            % large matrices
            n = size(T, 1);
            if nargin == 2
                T2 = diag(T * varargin{1} * T');
            else
                estimated_S = inv((T'*T)/(n));
                T2 = diag(T * estimated_S * T'); %#ok<MINV>
                if nargout > 1
                    varargout{1} = estimated_S;
                end
            end
        end
        
        function x = finv(p,v1,v2)
            %FINV   Inverse of the F cumulative distribution function.
            %   X=FINV(P,V1,V2) returns the inverse of the F distribution 
            %   function with V1 and V2 degrees of freedom, at the values in P.
            %   Copyright 1993-2004 The MathWorks, Inc.
            %   $Revision: 2.12.2.5 $  $Date: 2004/12/06 16:37:23 $

            % NOTE: This function is a pure copy and paste of the relevant parts of the Matlab statistical toolbox function "finv.m"
            % Please use the Statistical Toolbox function, if you have the toolbox

            % Guarantees:  0 < p < 1    :    The original stats toolbox function handles the case of p=0 and p=1
            %              v1 > 0
            %              v2 > 0
            z = mblvm.betainv(1 - p,v2/2,v1/2);
            x = (v2 ./ z - v2) ./ v1;
        end

        function x = betainv(p,a,b)
            %BETAINV Inverse of the beta cumulative distribution function (cdf).
            %   X = BETAINV(P,A,B) returns the inverse of the beta cdf with 
            %   parameters A and B at the values in P.
            %   Copyright 1993-2004 The MathWorks, Inc. 
            %   $Revision: 2.11.2.5 $  $Date: 2004/12/06 16:37:01 $

            %x = zeros(size(p));
            seps = sqrt(eps);

            % Newton's Method: permit no more than count_limit interations.
            count_limit = 100;
            count = 0;

            %   Use the mean as a starting guess. 
            xk = a ./ (a + b);

            % Move starting values away from the boundaries.
            xk(xk==0) = seps;
            xk(xk==1) = 1 - seps;
            h = ones(size(p));
            crit = seps;

            % Break out of the iteration loop for the following:
            %  1) The last update is very small (compared to x).
            %  2) The last update is very small (compared to 100*eps).
            %  3) There are more than 100 iterations. This should NEVER happen. 
            while(any(abs(h) > crit * abs(xk)) && max(abs(h)) > crit && count < count_limit), 
                count = count+1;    
                h = (mblvm.betacdf(xk,a,b) - p) ./ mblvm.betapdf(xk,a,b);
                xnew = xk - h;

            % Make sure that the values stay inside the bounds.
            % Initially, Newton's Method may take big steps.
                ksmall = find(xnew <= 0);
                klarge = find(xnew >= 1);
                if any(ksmall) || any(klarge)
                    xnew(ksmall) = xk(ksmall) /10;
                    xnew(klarge) = 1 - (1 - xk(klarge))/10;
                end
                xk = xnew;  
            end

            % Return the converged value(s).
            x = xk;

            if count==count_limit, 
                fprintf('\nWarning: BETAINV did not converge.\n');
                str = 'The last step was:  ';
                outstr = sprintf([str,'%13.8f\n'],max(h(:)));
                fprintf(outstr);
            end
        end

        function p = betacdf(x,a,b)
            %BETACDF Beta cumulative distribution function.
            %   P = BETACDF(X,A,B) returns the beta cumulative distribution
            %   function with parameters A and B at the values in X.
            %   Copyright 1993-2004 The MathWorks, Inc. 
            %   $Revision: 2.9.2.6 $  $Date: 2004/12/24 20:46:45 $

            % Initialize P to 0.
            p = zeros(size(x));
            p(a<=0 | b<=0) = NaN;

            % If is X >= 1 the cdf of X is 1. 
            p(x >= 1) = 1;

            k = find(x > 0 & x < 1 & a > 0 & b > 0);
            if any(k)
               p(k) = betainc(x(k),a(k),b(k));
            end
            % Make sure that round-off errors never make P greater than 1.
            p(p > 1) = 1;
        end

        function y = betapdf(x,a,b)
            %BETAPDF Beta probability density function.
            %   Y = BETAPDF(X,A,B) returns the beta probability density
            %   function with parameters A and B at the values in X.
            %   Copyright 1993-2004 The MathWorks, Inc.
            %   $Revision: 2.11.2.7 $  $Date: 2004/12/24 20:46:46 $

            % Return NaN for out of range parameters.
            a(a<=0) = NaN;
            b(b<=0) = NaN;

            % Out of range x could create a spurious NaN*i part to y, prevent that.
            % These entries will get set to zero later.
            xOutOfRange = (x<0) | (x>1);
            x(xOutOfRange) = .5;

            try
                % When a==1, the density has a limit of beta(a,b) at x==0, and
                % similarly when b==1 at x==1.  Force that, instead of 0*log(0) = NaN.
                warn = warning('off','MATLAB:log:logOfZero');
                logkerna = (a-1).*log(x);   logkerna(a==1 & x==0) = 0;
                logkernb = (b-1).*log(1-x); logkernb(b==1 & x==1) = 0;
                warning(warn);
                y = exp(logkerna+logkernb - betaln(a,b));
            catch %#ok<CTCH>
                warning(warn);
                error('stats:betapdf:InputSizeMismatch',...
                      'Non-scalar arguments must match in size.');
            end
            % Fill in for the out of range x values, but don't overwrite NaNs from nonpositive params.
            y(xOutOfRange & ~isnan(a) & ~isnan(b)) = 0;
        end
            
        function x = chi2inv(p, v)
            %CHI2INV Inverse of the chi-square cumulative distribution function (cdf).
            %   X = CHI2INV(P,V)  returns the inverse of the chi-square cdf with V  
            %   degrees of freedom at the values in P. The chi-square cdf with V 
            %   degrees of freedom, is the gamma cdf with parameters V/2 and 2.   

            % NOTE: This function is a pure copy and paste of the relevant parts of the Matlab statistical toolbox function "chi2inv.m"
            % Please use the Statistical Toolbox function, if you have the toolbox

            % Call the gamma inverse function. 

            % Guaratentees: % 0<p<1
            a = v/2;        % a>0
            b = 2;          % b>0
            %x = gaminv(p,v/2,2);

            % ==== Newton's Method to find a root of GAMCDF(X,A,B) = P ====
            maxiter = 500;
            iter = 0;

            % Choose a starting guess for q.  Use quantiles from a lognormal
            % distribution with the same mean (==a) and variance (==a) as G(a,1).
            loga = log(a);
            sigsq = log(1+a) - loga;
            mu = loga - 0.5 .* sigsq;
            q = exp(mu - sqrt(2.*sigsq).*erfcinv(2*p));
            h = ones(size(p));

            % Break out of the iteration loop when the relative size of the last step
            % is small for all elements of q.
            myeps = eps(class(p+a+b));
            reltol = myeps.^(3/4);
            dF = zeros(size(p));
            while any(abs(h(:)) > reltol*q(:))
                iter = iter + 1;
                if iter > maxiter
                    % Too many iterations.  This should not happen.
                    break
                end

                F = mblvm.gamcdf(q,a,1);
                f = max(mblvm.gampdf(q,a,1), realmin(class(p)));
                dF = F-p;
                h = dF ./ f;
                qnew = q - h;
                % Make sure that the current iterates stay positive.  When Newton's
                % Method suggests steps that lead to negative values, take a step
                % 9/10ths of the way to zero instead.
                ksmall = find(qnew <= 0);
                if ~isempty(ksmall)
                    qnew(ksmall) = q(ksmall) / 10;
                    h = q - qnew;
                end
                q = qnew;
            end

            badcdf = (isfinite(a) & abs(dF)>sqrt(myeps));
            if iter>maxiter || any(badcdf(:))   % too many iterations or cdf is too far off
                didnt = find(abs(h)>reltol*q | badcdf);
                didnt = didnt(1);
                if numel(a) == 1, abad = a; else abad = a(didnt); end
                if numel(b) == 1, bbad = b; else bbad = b(didnt); end
                if numel(p) == 1, pbad = p; else pbad = p(didnt); end
                warning('chi2inv:NoConvergence','Did not converge for a = %g, b = %g, p = %g.',abad,bbad,pbad);
            end
            x = q .* b;

        end
        
        function y = gampdf(x,a,b)
            %GAMPDF Gamma probability density function.
            %   Y = GAMPDF(X,A,B) returns the gamma probability density function with
            %   shape and scale parameters A and B, respectively, at the values in X.
            %   Copyright 1993-2004 The MathWorks, Inc.
            %   $Revision: 2.10.2.6 $  $Date: 2004/12/24 20:46:51 $

            % Return NaN for out of range parameters.
            a(a <= 0) = NaN;
            b(b <= 0) = NaN;

            z = x ./ b;
            % Negative data would create complex values, potentially creating
            % spurious NaNi's in other elements of y.  Map them to the far right
            % tail, which will be forced to zero.
            z(z < 0) = Inf;
            % Prevent LogOfZero warnings.
            warn = warning('off','MATLAB:log:logOfZero');
            u = (a - 1) .* log(z) - z - gammaln(a);
            warning(warn);

            % Get the correct limit for z == 0.
            u(z == 0 & a == 1) = 0;
            % These two cases work automatically
            %  u(z == 0 & a < 1) = Inf;
            %  u(z == 0 & a > 1) = -Inf;

            % Force a 0 for extreme right tail, instead of getting exp(Inf-Inf)==NaN
            u(z == Inf & isfinite(a)) = -Inf;
            % Force a 0 when a is infinite, instead of getting exp(Inf-Inf)==NaN
            u(z < Inf & a == Inf) = -Inf;
            y = exp(u) ./ b;

        end
        
        function p = gamcdf(x,a,b)
            %GAMCDF Gamma cumulative distribution function.
            %   P = GAMCDF(X,A,B) returns the gamma cumulative distribution function
            %   with shape and scale parameters A and B, respectively, at the values in X.
            %   Copyright 1993-2004 The MathWorks, Inc. 
            %   $Revision: 2.12.2.4 $  $Date: 2004/12/24 20:46:50 $
            
            x(x < 0) = 0;
            z = x ./ b;
            p = gammainc(z, a);
            p(z == Inf) = 1;
            
        end
        
        function x = tinv(p,v)
            %TINV   Inverse of Student's T cumulative distribution function (cdf).
            %   X=TINV(P,V) returns the inverse of Student's T cdf with V degrees 
            %   of freedom, at the values in P.
            %
            %   The size of X is the common size of P and V. A scalar input   
            %   functions as a constant matrix of the same size as the other input.    

            % NOTE: This function is a pure copy and paste of the relevant parts of the Matlab statistical toolbox function "finv.m"
            % Please use the Statistical Toolbox function, if you have the toolbox

            %   References:
            %      [1]  M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
            %      Functions", Government Printing Office, 1964, 26.6.2

            % Initialize Y to zero, or NaN for invalid d.f.
            if isa(p,'single') || isa(v,'single')
                x = NaN(size(p),'single');
            else
                x = NaN(size(p));
            end

            % The inverse cdf of 0 is -Inf, and the inverse cdf of 1 is Inf.
            x(p==0 & v > 0) = -Inf;
            x(p==1 & v > 0) = Inf;

            k0 = (0<p & p<1) & (v > 0);

            % Invert the Cauchy distribution explicitly
            k = find(k0 & (v == 1));
            if any(k)
              x(k) = tan(pi * (p(k) - 0.5));
            end

            % For small d.f., call betainv which uses Newton's method
            k = find(k0 & (v < 1000));
            if any(k)
                q = p(k) - .5;
                df = v(k);
                t = (abs(q) < .25);
                z = zeros(size(q),class(x));
                oneminusz = zeros(size(q),class(x));
                if any(t)
                    % for z close to 1, compute 1-z directly to avoid roundoff
                    oneminusz(t) = betainv(2.*abs(q(t)),0.5,df(t)/2);
                    z(t) = 1 - oneminusz(t);
                end
                if any(~t)
                    z(~t) = mblvm.betainv(1-2.*abs(q(~t)),df(~t)/2,0.5);
                    oneminusz(~t) = 1 - z(~t);
                end
                x(k) = sign(q) .* sqrt(df .* (oneminusz./z));
            end

            % For large d.f., use Abramowitz & Stegun formula 26.7.5
            % k = find(p>0 & p<1 & ~isnan(x) & v >= 1000);
            k = find(k0 & (v >= 1000));
            if any(k)
               xn = mblvm.norminv(p(k));
               df = v(k);
               x(k) = xn + (xn.^3+xn)./(4*df) + ...
                       (5*xn.^5+16.*xn.^3+3*xn)./(96*df.^2) + ...
                       (3*xn.^7+19*xn.^5+17*xn.^3-15*xn)./(384*df.^3) +...
                       (79*xn.^9+776*xn.^7+1482*xn.^5-1920*xn.^3-945*xn)./(92160*df.^4);
            end
        end
        
        function [x,xlo,xup] = norminv(p,mu,sigma,pcov,alpha)
            %NORMINV Inverse of the normal cumulative distribution function (cdf).
            %   X = NORMINV(P,MU,SIGMA) returns the inverse cdf for the normal
            %   distribution with mean MU and standard deviation SIGMA, evaluated at
            %   the values in P.  The size of X is the common size of the input
            %   arguments.  A scalar input functions as a constant matrix of the same
            %   size as the other inputs.
            %
            %   Default values for MU and SIGMA are 0 and 1, respectively.
            %
            %   [X,XLO,XUP] = NORMINV(P,MU,SIGMA,PCOV,ALPHA) produces confidence bounds
            %   for X when the input parameters MU and SIGMA are estimates.  PCOV is a
            %   2-by-2 matrix containing the covariance matrix of the estimated parameters.
            %   ALPHA has a default value of 0.05, and specifies 100*(1-ALPHA)% confidence
            %   bounds.  XLO and XUP are arrays of the same size as X containing the lower
            %   and upper confidence bounds.
            %
            %   See also ERFINV, ERFCINV, NORMCDF, NORMFIT, NORMLIKE, NORMPDF,
            %            NORMRND, NORMSTAT.

            %   References:
            %      [1] Abramowitz, M. and Stegun, I.A. (1964) Handbook of Mathematical
            %          Functions, Dover, New York, 1046pp., sections 7.1, 26.2.
            %      [2] Evans, M., Hastings, N., and Peacock, B. (1993) Statistical
            %          Distributions, 2nd ed., Wiley, 170pp.

            %   Copyright 1993-2004 The MathWorks, Inc. 
            %   $Revision: 2.16.4.2 $  $Date: 2004/08/20 20:06:03 $

            if nargin < 2
                mu = 0;
            end
            if nargin < 3
                sigma = 1;
            end

            % More checking if we need to compute confidence bounds.
            if nargout>2
               if nargin<5
                  alpha = 0.05;
               end
            end

            % Return NaN for out of range parameters or probabilities.
            sigma(sigma <= 0) = NaN;
            p(p < 0 | 1 < p) = NaN;

            x0 = -sqrt(2).*erfcinv(2*p);
            try %#ok<TRYNC>
                x = sigma.*x0 + mu;
            end

            % Compute confidence bounds if requested.
            if nargout>=2
               xvar = pcov(1,1) + 2*pcov(1,2)*x0 + pcov(2,2)*x0.^2;
               normz = -norminv(alpha/2);
               halfwidth = normz * sqrt(xvar);
               xlo = x - halfwidth;
               xup = x + halfwidth;
            end

        end
        
        function limits = spe_limits(values, levels, ncol)
            % Calculates the SPE limit(s) at the given ``levels`` [0, 1]
            % where SPE was calculated over ``ncol`` entries.
            %
            % NOTE: our SPE values = sqrt( e*e^T / K )
            %       where e = row vector of residuals from the model
            %       and K = number of columns (length of vector e)
            %
            % The SPE limit is defined for a vector ``e``, so we need to undo
            % our SPE transformation, calculate the SPE limit, then transform
            % the SPE limit back.  That's why this function requires ``ncol``,
            % which is the same as K in the above equation.
            
            values = values.^2 .* ncol;
            if all(values) < sqrt(eps)
                limits = 0.0;
                return
            end
            var_SPE = var(values);
            avg_SPE = mean(values);
            chi2_mult = var_SPE/(2.0 * avg_SPE);
            chi2_DOF = (2.0*avg_SPE^2)/var_SPE;
            limits = chi2_mult * mblvm.chi2inv(levels, chi2_DOF);
            
            limits = sqrt(limits ./ ncol);

            % For batch blocks: calculate instantaneous SPE using a window
            % of width = 2w+1 (default value for w=2).
            % This allows for (2w+1)*N observations to be used to calculate
            % the SPE limit, instead of just the usual N observations.
            %
            % Also for batch systems:
            % low values of chi2_DOF: large variability of only a few variables
            % high values: more stable periods: all k's contribute
        end

        function limits = T2_limits(scores, levels, A)
            % Calculates the T2 limits from columns of ``scores`` at
            % significance ``levels``.  Uses from 1 to ``A`` columns in
            % scores.
            
            % TODO(KGD): take the covariance in ``scores`` into account to
            % calculate the elliptical confidence interval for T2.
            
            n = size(scores, 1);
            mult = A*(n-1)*(n+1)/(n*(n-A));
            f_limit = mblvm.finv(levels, A, n - A);
            limits = mult * f_limit;
        end
         
        function limits = score_limits(score_column, levels)
            % Assume that scores are from a two-sided t-distribution with N-1
            % degrees of freedom.  Based on the central limit theorem.
            % ``score_limits`` can be a matrix, in which case ``limits`` are
            % the confidence limits for each column.
            n = size(score_column, 1);
            alpha = (1-levels) ./ 2.0;
            n_ppf = mblvm.tinv(1-alpha, n-1);            

            % NORMAL DISTRIBUTION ASSUMPTION
            % 
            % n_ppf = mblvm.norminv(1-alpha);
            
            limits = n_ppf .* std(score_column, 0, 1);
        end

    end % end: methods (sealed and static)
    
    methods (Static=true)
        function [hF, hHead, hFoot] = add_figure()
            % Adds a new figure
            background_colour = [1, 1, 1];
            font_size = 14;
            hF = figure('Color', background_colour);
            set(hF, 'ToolBar', 'figure')
            units = get(hF, 'Units');
            set(hF, 'units', 'Pixels');
             
            screen = get(0,'ScreenSize');   
            fPos = get(hF, 'position');
            fPos(1) = round(0.05*screen(3));
            fPos(2) = round(0.1*screen(4));
            fPos(3) = 0.90*screen(3);
            fPos(4) = 0.80*screen(4);
            set(hF, 'Position', fPos);
  
            Txt.Interruptible = 'off';  % Text fields different to default values
            Txt.BusyAction    = 'queue';
            Txt.Style         = 'text';
            Txt.Units         = 'normalized';
            Txt.HorizontalAli = 'center';
            Txt.Background    = background_colour;
            Txt.FontSize      = font_size;
            Txt.Parent        = hF;
            set(hF, 'Units', 'normalized')
            
            hHead = uicontrol(Txt, ...
                             'Position',[0, 0, eps, eps], ...  %'Position',[0.02, 0.95, 0.96 0.04], ...
                             'ForegroundColor',[0 0 0], 'String', '');
            %set(hHead, 'Units', 'Pixels')
            %p = get(hHead, 'Position')
            Txt.FontSize = 10;
            hFoot = uicontrol(Txt, ...
                             'Position',[0.02, 0.005, 0.96 0.04], ...
                             'ForegroundColor',[0 0 0], 'String', '', ...
                             'HorizontalAlignment', 'left');
            set(hF, 'units', units);
        end
        
        function add_plot_window_title(hHeaders, text_to_add)
            for k = 1:numel(hHeaders)
                hF = get(hHeaders(k), 'Parent');
                set(hF, 'Name', text_to_add);
            end
        end
        
        function add_plot_footers(hFooters, footer_string)
            % Convert the cell string to a long char string
            foot = footer_string{1};
            for j = 2:numel(footer_string)
                foot = [foot, footer_string{j}]; %#ok<AGROW>
            end
            for k = 1:numel(hFooters)
                set(hFooters(k), 'String', footer_string)
            end
        end

    end % end methods (static)
    
end % end classdef

%-------- Helper functions (usually used in 2 or more places). May NOT change ``self``
function add_text_labels(hA, x, y, labels, popt)
    % Adds text labels to axis ``hA`` at the ``x`` and ``y`` coordinates
    if popt.show_labels
        extent = axis(hA);
        delta_x = 0.01*diff(extent(1:2));
        delta_y = 0.01*diff(extent(3:4));
        for n = 1:numel(x)
            text(x+delta_x, y+delta_y, labels(n), 'parent', hA)
        end
    end
end

function h = plot_score_scatterplot(t_h, t_v, block, popt)
    % Plots the t_h vs t_v score scatterplot with the confidence limit (95%).
    % Uses component ``th`` on the horizontal axis and ``tv`` on the vertical
    % axis.
    
    % TODO(KGD): handle the case with tilted ellipse: i.e. correlated scores
    
    
    h = plot(t_h, t_v, 'k.'); 
    hold on
    
    [x95, y95] = ellipse_coordinates(block.S(ah), block.S(av), block.lim.T2(end), 100);
    plot(x95, y95, 'r--', 'linewidth', 1)
    title('Score target plot')
    grid on
    %axis equal 
    xlabel(['t_', num2str(ah)])
    ylabel(['t_', num2str(av)])    
    extent = axis;
    hd = plot([0, 0], [-1E10, 1E10], 'k', 'linewidth', 2);
    set(hd, 'tag', 'vline', 'HandleVisibility', 'off');
    hd = plot([-1E50, 1E50], [0, 0], 'k', 'linewidth', 2);
    set(hd, 'tag', 'hline', 'HandleVisibility', 'off');
    xlim(extent(1:2))
    ylim(extent(3:4)) 
    
    add_text_labels(h, t_h, t_v, block.labels{1,1}, popt)
    
    
end
        
function [hA, hHeaders, hFooters, title_str] = plot_scores(self, popt)
    hA = 0;
    hHeaders = [];
    hFooters = [];
    title_str = 'Score plots';
    popt.footer_string = {title_str, popt.footer_string{:}};
    % [hF, hHead, hFoot] = self.add_figure(); %#ok<ASGLU>
    %hHeaders(end+1) = hHead; %#ok<*AGROW>
    %hFooters(end+1) = hFoot;

%     
%     if blockX.A == 1        
%         plot_score_lineplot(blockX, 1, show_labels);
%    
%     elseif blockX.A == 2
%         nrow = 1;
%         ncol = 2;
%         subplot(nrow, ncol, ax);
%         plot_score_scatterplot(blockX, 1, 2, show_labels);
%         ax = ax + 1;
%         
%         % t1-t2 scatter plot with ellipse
%         subplot(nrow, ncol, ax);
%         plot_Hotellings_T2_lineplot(blockX, true, show_labels);
%     elseif blockX.A >= 3
%         nrow = 2;
%         ncol = 2;
%         
%         subplot(nrow, ncol, ax);
%         plot_score_scatterplot(blockX, 1, 2, show_labels);
%         ax = ax + 1;
%         subplot(nrow, ncol, ax);
%         plot_score_scatterplot(blockX, 3, 2, show_labels);
%         ax = ax + 1;
%         subplot(nrow, ncol, ax);
%         plot_score_scatterplot(blockX, 1, 3, show_labels);
%         ax = ax + 1;
%         subplot(nrow, ncol, ax);
%         plot_Hotellings_T2_lineplot(blockX, true, show_labels);
%     end    
end % ``plot_scores``

function [hA, hHeaders, hFooters, title_str] = plot_loadings(self, popt)
    hA = 0;
    hHeaders = [];
    hFooters = [];
    title_str = 'Loading plots';
    popt.footer_string = {title_str, popt.footer_string{:}};
    
    disp('Plot the scores')
    popt
end % ``plot_loadings``

function [hA, hHeaders, hFooters, title_str] = plot_T2(self, popt)
    hA = 0;
    hHeaders = [];
    hFooters = [];
    title_str = 'Hotelling''s T2 plot';
    popt.footer_string = {title_str, popt.footer_string{:}};
    
    disp('Plot HT2')
    popt
end % ``plot_T2``

function [hA, hHeaders, hFooters, title_str] = plot_SPE(self, popt)
    hA = 0;
    hHeaders = [];
    hFooters = [];
    title_str = 'SPE plots';
    popt.footer_string = {title_str, popt.footer_string{:}};
    
    disp('Plot SPE')
    popt
end % ``plot_SPE``

function [hA, hHeaders, hFooters, title_str] = plot_R2(self, popt)
    hA = 0;
    hHeaders = [];
    hFooters = [];
    title_str = 'R2 plots';
    popt.footer_string = {title_str, popt.footer_string{:}};
    
    disp('Plot R2')
    popt
end % ``plot_R2``

function [hA, hHeaders, hFooters, title_str] = plot_summary(self, popt)
    hA = 0;
    hHeaders = [];
    hFooters = [];
    title_str = 'Summary plots';
    popt.footer_string = {title_str, popt.footer_string{:}};
    
    disp('Plot Summary')
    popt
end % ``plot_summary``