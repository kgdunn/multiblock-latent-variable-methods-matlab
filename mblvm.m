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
        %C = cell({});       % Block loadings (for PLS only)
        %U = cell({});       % Block scores (for PLS only) 
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
                self.has_missing = self.has_missing;            
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
            self.super.lim.SPE(a) = self.spe_limits(self.super.SPE(:,a), siglevel);
            self.super.lim.T2(a) = self.T2_limits(self.super.T(:,1:a), siglevel, a);
            self.super.lim.t(a) = self.score_limits(self.super.T(:, a), siglevel);
            
            for b = 1:self.B
                self.lim{b}.SPE(a) = self.spe_limits(self.stats{b}.SPE(:,a), siglevel);
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
            % Avoid this step for now
            return;
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
            
            for b = 1:numel(batch_blocks)
                b_id = batch_blocks(b);
                self.stats{b}.T_j = zeros(self.N, self.A, self.blocks{b_id}.J);
                J_time = self.blocks{b_id}.J;                
            end
            self.super.T2_j = zeros(self.N, J_time);
            self.super.SPE_j = zeros(self.N, J_time);
            SPE_j_temp = zeros(self.N, J_time);
            
            show_progress = self.opt.show_progress;
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
                fprintf('.')
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
                            
                            self.super.T2_j(n,j) = out.stats.super.T2;
                            self.super.SPE_j(n,j) = out.stats.super.SPE;  % out.stats.SPE(1, end); % use only the last component
                            self.stats{b}.T_j(n, :, j) = out.T{b};
                            SPE_j_temp(n,j) = ssq(out.newb{b}.data(1, idx_beg:idx_end));
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
                        
                        
                        N_lim = self.blocks{b}.N;
                        for a = 1:self.A
                            mult = a*(N_lim-1)*(N_lim+1)/(N_lim*(N_lim-a));
                            limit = self.finv(siglevel, a, N_lim-(self.A));
                            self.lim{b}.T2_j(:,a) = mult * limit;
                            % This value should agree with batch.lim.T2(:,a)
                            % TODO(KGD): slight discrepancy in batch SBR dataset
                        end

                        % Calculate SPE time-varying limits
                        % Our SPE definition = e'e, where e is the sum of 
                        % squares in the row, not counting missing data.  
                        % 
                        % Apply smoothing window here: see Nomikos thesis, p 66.
                        % ``w`` should be a function of how "jumpy" the SPE plot
                        % looks.  Right now, we set ``w`` proportional
                        % to the number of time samples in a batch
                        w = max(1, ceil(0.012/2 * self.blocks{b}.J));
                        for j = 1:self.blocks{b}.J
                            start_idx = max(1, j-w);
                            end_idx = min(self.blocks{b}.J, j+w);
                            SPE_values = SPE_j_temp(:, start_idx:end_idx);
                            self.lim{b}.SPE_j(j) = self.spe_limits(SPE_values, siglevel);
                        end   
                        % N x J matrix assigned
                        self.stats{b}.SPE_j = SPE_j_temp;
                        %self.stats{b}.SPE_j = zeroexp([dblock.N, dblock.J], self.stats{b}.SPE_j, true);
                        
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
            state.Y_pred = ones(Nnew, self.M) .* NaN;
            
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
            %self.C = cell(1,nb);
            %self.U = cell(1,nb);
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
                    self.stats{b}.R2Xk_a = [];  % is it cumulative ?
                    self.stats{b}.col_ssq_prior = [];
                    self.stats{b}.R2Xb_a = [];
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
                self.stats{b}.R2Xb_a = zeroexp([1, A], self.stats{b}.R2Xb_a);
                
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
                
        function get_contribution(self, varargin)
            hP = varargin{1};
            series = getappdata(get(hP.hMarker,'Parent'), 'SeriesData');
            idx = get(hP.hMarker, 'UserData');
            if hP.c_block == 0
                error('Cannot calculate contributions for superblock items');
            end
            contrib = [];
            switch series.y_type{1}
                case 'SPE'
                    block_index = self.b_iter(hP.c_block);
                    contrib = self.data(idx, block_index);
                    contrib = sign(contrib) .* contrib .^2;
                    labels= self.blocks{hP.c_block}.labels{2};
                    
                    
                case 'Scores'                    
                    pp_data = self.blocks{hP.c_block}.data(idx,:);
                    contrib = zeros(size(pp_data));
                    contrib = contrib(:);
                    
                    % This shouldn't branch like this: put this in each class
                    if strcmp(self.model_type, 'PCA') || strcmp(self.model_type, 'MB-PCA')
                        weights = self.P{hP.c_block};
                    elseif strcmp(self.model_type, 'PLS') || strcmp(self.model_type, 'MB-PLS')
                        temp_W = self.W{hP.c_block};
                        temp_P = self.P{hP.c_block};
                        weights = temp_W*inv(temp_P'*temp_W);  % W-star: is this correct for blocks: non-orthogonal scores?
                    end
                    scores = self.T{hP.c_block}(idx, :);
                    score_variance = std(self.T{hP.c_block});
                    if series.x_num > 0
                        contrib = contrib + ...
                                  abs(scores(series.x_num)./score_variance(series.x_num)) .* ...
                                  abs(weights(:,series.x_num) .* pp_data(:)) .* sign(pp_data(:));
                    end
                    if series.y_num > 0
                        contrib = contrib + ...
                                  abs(scores(series.y_num)./score_variance(series.y_num)) .* ...
                                  abs(weights(:,series.y_num) .* pp_data(:)) .* sign(pp_data(:));
                    end
                    labels= self.blocks{hP.c_block}.labels{2};
            end
            
            figure('Color', 'white')
            hAx = axes;
            if isempty(contrib)
                text(0.5, 0.5, 'Contributions not available for this plot type ... yet', ...
                    'HorizontalAlignment', 'center', 'FontSize', 12);
                set(hAx,'Visible', 'off')
                return
            end
            hBar = bar(contrib);
            if isa(hP.model.blocks{hP.c_block}, 'block_batch')
                hP.annotate_batch_trajectory_plots(hAx, hBar, hP.model.blocks{hP.c_block})
            else
                if not(isempty(labels))
                    set(hAx, 'XTickLabel', labels)
                end
            end
                
        end
        
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
            
            % IT IS WEIRD THAT the mblvm class is telling how to use the 
            % ``lvmplot`` class with all the steps involved.  Or is it?
            % Should the ``lvmplot`` class be merely a convenient plot
            % wrapper?
            
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
            
            if nargin == 1
                h = lvmplot(self);
                for i=1:nargout
                    varargout{i} = h;
                end
                return
            elseif nargin==3
                show_what = varargin{2};
                h = lvmplot(self, show_what);                
                h.new_figure();  % Start a new figure window
                
                %h = varargin{1};
                %show_what = varargin{2};
                %h.clear_figure();
                %h.ptype = show_what;
            else
                show_what = varargin{1};
                h = lvmplot(self, show_what);                
                h.new_figure();  % Start a new figure window
            end
            
            switch lower(show_what)
                case 'scores'
                    basic_plot__scores(h);
                case 'loadings'
                    basic_plot__loadings(h);
                case 'weights'
                    basic_plot__weights(h);
                case 'spe'
                    basic_plot__spe(h);                    
                case 'predictions'
                    basic_plot__predictions(h);
                case 'coefficient'
                    basic_plot__coefficient(h);
                case 'vip'
                    basic_plot__VIP(h);
                case 'r2-variable'
                    basic_plot_R2_variable(h);
                case 'r2-component'
                    basic_plot_R2_component(h);
                case 'r2-y-variable'
                    basic_plot_R2_Y_variable(h)
                case 'weights-batch'
                    batch_plot__weights(h);
                    
            end
            h.update_all_plots();
            h.update_annotations();
            
            set(h.hF, 'Visible', 'on')
%             for i=1:nargout
%                 varargout{i} = h;
%             end

%             self.add_plot_footers(hFooters, popt.footer_string);
%             self.add_plot_window_title(hHeaders, title_str)
        end % ``plot``
        
        function out = register_plots(self)
            % Register which variables in ``self`` are plottable
            % Subclasses get to over ride the registration afterwards by
            % defining registrations in self.registers_plots_post()
            out = [];
            
            % These plots are common for PCA and PLS models
            
            % plt.name      : name in the drop down
            % plt.weight    : ordering in drop down menu: higher values first
            % plt.dim       : dimensionality: 0=component, 1=rows, 2=columns
            % plt.callback  : function that will show the plot
            % plt.more_text : text that goes between two dropdowns "for component"
            % plt.more_type : tells what to put in the second dropdown
            % plt.more_block: which block is the more_type variable referring to
            % plt.annotate  : callback to add annotations to that axis (limits)
            
            plt = struct;  
            
            % Dimension 0 (component) plots
            % =========================            
            plt.name = 'Order';
            plt.weight = 0;
            plt.dim = 0;
            plt.more_text = '(component order)';
            plt.more_type = '<order>';
            plt.more_block = '';
            plt.annotate = @self.order_dim0_annotate;
            plt.callback = @self.order_dim0_plot;
            out = [out; plt];            
            
            plt.name = 'R2 (per component)';
            plt.weight = 60;
            plt.dim = 0;
            plt.more_text = '';
            plt.more_type = '<model>';
            plt.more_block = '';
            plt.annotate = '';
            plt.callback = @self.R2_component_plot;
            out = [out; plt];
                        
            % plt.name = 'Eigenvalues'; 
            
            % Dimension 1 (rows) plots
            % =========================            
            plt.name = 'Order';
            plt.weight = 0;
            plt.dim = 1;
            plt.more_text = '(row order)';
            plt.more_type = '<order>';
            plt.more_block = '';
            plt.annotate = @self.order_dim1_annotate;
            plt.callback = @self.order_dim1_plot;
            out = [out; plt];            
            
            plt.name = 'Scores';   % t-scores
            plt.weight = 60;
            plt.dim = 1;
            plt.more_text = ': of component';
            plt.more_type = 'a';
            plt.more_block = '';
            plt.callback = @self.score_plot;
            plt.annotate = @self.score_limits_annotate;            
            out = [out; plt];            
            
            plt.name = 'Hot T2';  % Hotelling's T2
            plt.weight = 40;
            plt.dim = 1;
            plt.more_text = 'using  components';
            plt.more_type = '1:A';
            plt.more_block = '';
            plt.callback = @self.hot_T2_plot;
            plt.annotate = @self.hot_T2_limits_annotate;
            out = [out; plt];
            
            plt.name = 'SPE';
            plt.weight = 50;
            plt.dim = 1;
            plt.more_text = 'using  components';
            plt.more_type = '1:A';
            plt.more_block = '';
            plt.callback = @self.spe_plot;
            plt.annotate = @self.spe_limits_annotate;
            out = [out; plt];
            
            % Dimension 2 (columns) plots
            % ============================     
            plt.name = 'Order';
            plt.weight = 0;
            plt.dim = 2;
            plt.more_text = '(column order)';
            plt.more_type = '<order>';
            plt.more_block = '';
            plt.annotate = @self.order_dim2_annotate;
            plt.callback = @self.order_dim2_plot;
            out = [out; plt];            
            
            plt.name = 'Loadings';
            plt.weight = 20;
            plt.dim = 2;
            plt.more_text = ': of component';
            plt.more_type = 'a';
            plt.more_block = '';
            plt.callback = @self.loadings_plot;
            plt.annotate = @self.loadings_plot_annotate;
            out = [out; plt];
            
            plt.name = 'VIP';
            plt.weight = 40;
            plt.dim = 2;
            plt.more_text = 'using  components';
            plt.more_type = '1:A';
            plt.more_block = '';
            plt.callback = @self.VIP_plot;
            plt.annotate = @self.VIP_limits_annotate;
            out = [out; plt];
            
            % KGD: why is this under the mblvm class: shouldn't it be in PLS?
            plt.name = 'Coefficient';
            plt.weight = 40;
            plt.dim = 2;
            plt.more_text = 'using  components';
            plt.more_type = '1:A';
            plt.more_block = '';
            plt.callback = @self.coefficient_plot;
            plt.annotate = @self.coefficient_annotate;
            out = [out; plt];
            
            plt.name = 'R2 (per variable)';
            plt.weight = 50;
            plt.dim = 2;
            plt.more_text = 'using  components';
            plt.more_type = '1:A';
            plt.more_block = '';
            plt.callback = @self.R2_per_variable_plot;
            plt.annotate = @self.R2_per_variable_plot_annotate;
            out = [out; plt];
            
            % plt.name = 'Centering';
            % plt.name = 'Scaling';
             
            extra = register_plots_post(self);
            out = [out; extra];
        end % ``register_plots``
        
        function out = get_labels(self, dim, req_block)
            % Finds the labels for the given ``dim``ension and the optional
            % ``block`` number
            out = [];
            if nargin == 1
                req_block = 0;
            end
            
            if dim == 1 && req_block == 0 
            % search through all blocks to find these labels
                for b = 1:self.B
                    if ~isempty(self.blocks{b}.labels{dim})
                        out = self.blocks{b}.labels{dim};
                        return
                    end
                end
            else
                if req_block ~= 0
                    
                    out = self.blocks{req_block}.labels{dim};
                else
                    block_names = cell(self.B, 1);            
                    for b = 1:self.B
                        block_names{b} = self.blocks{b}.name;
                    end
                    out = block_names;
                end
            end            
        end
    end % end: methods (sealed)
    
    % Subclasses must redefine these methods
    methods (Abstract=true)
        preprocess_extra(self)   % extra preprocessing steps that subclass does
        calc_model(self, A)      % fit the model to data in ``self``
        limits_subclass(self)    % post model calculation: fit additional limits
        apply_model(self, other) % applies the model to ``new`` data
        expand_storage(self, A)  % expands the storage to accomodate ``A`` components
        summary(self)            % show a text-based summary of ``self``
        register_plots_post(self)% register which variables are plottable      
    end % end: methods (abstract)
    
    % Subclasses may not redefine these methods
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
        
        function limits = spe_limits(values, levels)
            % Calculates the SPE limit(s) at the given ``levels`` [0, 1]
            %
            % NOTE: our SPE values = e*e^T where e = row vector of residuals
            %       from the model
           
            % Then inputs were e'e (not sqrt(e'e/K))
            values = values(:);
            if all(values) < sqrt(eps)
                limits = 0.0;
                return
            end
            var_SPE = var(values);
            avg_SPE = mean(values);
            chi2_mult = var_SPE/(2.0 * avg_SPE);
            chi2_DOF = (2.0*avg_SPE^2)/var_SPE;
            limits = chi2_mult * mblvm.chi2inv(levels, chi2_DOF);
            
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
        function order_dim0_plot(hP, varargin)            
            ax = hP.gca();
            n = hP.model.A;
            hP.set_data(ax, 1:n, []);
        end        
        function order_dim0_annotate(hP, varargin)
            ax = hP.gca();
            xlabel(ax, 'Component order')
        end
        
        function order_dim1_plot(hP, varargin)
            ax = hP.gca();
            if hP.c_block > 0
                n = hP.model.blocks{hP.c_block}.shape(hP.dim);
            else
                n = size(hP.model.super.T, 1);
            end
            hP.set_data(ax, 1:n, []);
        end        
        function order_dim1_annotate(hP, varargin)
            ax = hP.gca();
            xlabel(ax, 'Row order')
        end
        
        function order_dim2_plot(hP, varargin)
            ax = hP.gca();
            if hP.c_block > 0
                n = hP.model.blocks{hP.c_block}.shape(hP.dim);
            else
                n = size(hP.model.super.T, 2);
            end
            hP.set_data(ax, 1:n, []);
        end
        function order_dim2_annotate(hP, varargin)
            ax = hP.gca();
            xlabel(ax, 'Column order', 'FontSize', 15);
        end
        
        function [scores_h, scores_v] = get_score_data(hP, series)
            % Correctly fetches the score data for the current block and dropdowns
            t_h = series.x_num;
            t_v = series.y_num;
            block = hP.c_block;
            scores_h.data = [];
            scores_h.limit = [];
            scores_v.data = [];
            scores_v.limit = [];
            
            if t_h <= 0
                if block == 0
                    scores_v.data = hP.model.super.T(:, t_v);
                    scores_v.limit = hP.model.super.lim.t(t_v);
                else
                    scores_v.data = hP.model.T{block}(:, t_v);
                    scores_v.limit = hP.model.lim{block}.t(t_v);
                end
            % Joint plots, i.e. add Hotelling's T2 ellipse
            else
                if block == 0
                    scores = hP.model.super.T(:,[t_h t_v]); 
                    scores_h.limit = hP.model.super.lim.T2(2);  % TODO(KGD): which limit to use?
                else
                    scores = hP.model.T{block}(:,[t_h t_v]);                   
                    scores_h.limit = hP.model.lim{block}.T2(2); % TODO(KGD): which limit to use?
                end
                scores_h.data = scores(:,1);                
                scores_v.data = scores(:,2);
                scores_v.limit = scores_h.limit;
            end
            
        end        
        function score_plot(hP, series)
            % Score plots for overall or block scores
            
            ax = hP.gca();
            [scores_h, scores_v] = hP.model.get_score_data(hP, series);
            
            
            % Line plot of the vertical entity
            if series.x_num <= 0
                hPlot = hP.set_data(ax, scores_h.data, scores_v.data);
                set(hPlot, 'LineStyle', '-', 'Marker', '.', 'Color', [0, 0, 0])
                %set(ax, 'YLim'
                return
            end
            
            % Scatter plot of the scores            
            hPlot = hP.set_data(ax, scores_h.data, scores_v.data);
            set(hPlot, 'LineStyle', 'none', 'Marker', '.', 'Color', [0, 0, 0])
            
        end       
        function score_limits_annotate(hP, series)
            ax = hP.gca();            
            t_h = series.x_num;
            t_v = series.y_num;
            
            % Scatter plot of the scores
            [scores_h, scores_v] = hP.model.get_score_data(hP, series);
            
            % Add ellipse
            if t_h > 0 && t_v > 0
                set(ax, 'NextPlot', 'add')
                ellipse_coords = ellipse_coordinates([scores_h.data scores_v.data], scores_h.limit);
                plot(ax, ellipse_coords(:,1), ellipse_coords(:,2), 'r--', 'linewidth', 1)            
                title(ax, 'Score plot')
                grid on
                xlabel(ax, ['t_', num2str(t_h)])
                ylabel(ax, ['t_', num2str(t_v)])
                
                % Create a balanced view by making the limits symmetrical
                extent = axis;  
                extent(1:2) = max(abs(extent(1:2))) .* [-1 1];
                extent(3:4) = max(abs(extent(3:4))) .* [-1 1];
                set(ax, 'XLim', extent(1:2), 'YLim', extent(3:4))
                
            % Add univariate score limits
            elseif t_v > 0
                set(ax, 'NextPlot', 'add')
                extent = axis;  
                hd = plot([-1E50, 1E50], [scores_v.limit, scores_v.limit], 'r-', 'linewidth', 1);
                set(hd, 'tag', 'hLimit', 'HandleVisibility', 'on');
                hd = plot([-1E50, 1E50], [-scores_v.limit, -scores_v.limit], 'r-', 'linewidth', 1);
                set(hd, 'tag', 'hLimit', 'HandleVisibility', 'on');
                title(ax, 'Score line plot')
                grid on
                ylabel(ax, ['t_', num2str(t_v)])
                xlim(extent(1:2))
                ylim(extent(3:4)) 
                
            end
                        
            hPlot = findobj(ax, 'Tag', 'lvmplot_series');
            labels = hP.model.get_labels(hP.dim, hP.c_block);
            hP.label_scatterplot(hPlot, labels);
        end
        
        function hot_T2_plot(hP, series)
            ax = hP.gca();
            block = hP.c_block;
            if strcmpi(series.current, 'x')
                t_idx = series.x_num;
            elseif strcmpi(series.current, 'y')
                t_idx = series.y_num;
            end       
            
            if block == 0
                plotdata = hP.model.super.T2(:, t_idx);
            else
                plotdata = hP.model.stats{block}.T2(:, t_idx);
            end
            
            if strcmpi(series.current, 'x')
                hPlot = hP.set_data(ax, plotdata, []);
            elseif strcmpi(series.current, 'y')
                hPlot = hP.set_data(ax, [], plotdata);
            end
            set(hPlot, 'LineStyle', '-', 'Marker', '.', 'Color', [0, 0, 0])
        end
        function hot_T2_limits_annotate(hP, series)
            ax = hP.gca();
            block = hP.c_block;
            if strcmpi(series.current, 'x')
                t_idx = series.x_num;
            elseif strcmpi(series.current, 'y')
                t_idx = series.y_num;
            end                
            
            if block == 0
                limit = hP.model.super.lim.T2(t_idx);
            else
                limit = hP.model.lim{block}.T2(t_idx);
            end
            
            % Which axis to add the limit to?
            set(ax, 'NextPlot', 'add')
            extent = axis;
            if strcmpi(series.current, 'x')
                hd = plot([limit, limit], [-1E10, 1E10], 'r-', 'linewidth', 1);
                xlabel(ax ,'Hotelling''s T^2')
            elseif strcmpi(series.current, 'y')
                hd = plot([-1E10, 1E10], [limit, limit], 'r-', 'linewidth', 1);                
                ylabel(ax, 'Hotelling''s T^2')
            end
            set(hd, 'tag', 'hLimit', 'HandleVisibility', 'on');
            
            if series.x_num < 0 && series.y_num > 0
                title(ax, 'Hotelling''s T^2 plot')
            end
            grid on            
            xlim(extent(1:2))
            ylim(extent(3:4)) 
            
            hPlot = findobj(ax, 'Tag', 'lvmplot_series');
            labels = hP.model.get_labels(hP.dim, hP.c_block);
            hP.label_scatterplot(hPlot, labels);
        end
        
        function spe_data = get_spe_data(hP, series)
            block = hP.c_block;
            if strcmpi(series.current, 'x')
                idx = series.x_num;
            elseif strcmpi(series.current, 'y')
                idx = series.y_num;
            end       
            
            if block == 0
                spe_data = hP.model.super.SPE(:, idx);
            else
                spe_data = hP.model.stats{block}.SPE(:, idx);
            end
        end
        function spe_plot(hP, series)
            ax = hP.gca();
            spe_data = hP.model.get_spe_data(hP, series);
            if strcmpi(series.current, 'x')
                hPlot = hP.set_data(ax, spe_data, []);
            elseif strcmpi(series.current, 'y')
                hPlot = hP.set_data(ax, [], spe_data);
            end
            set(hPlot, 'LineStyle', '-', 'Marker', '.', 'Color', [0, 0, 0])
        end
        function spe_limits_annotate(hP, series)
            ax = hP.gca();
            block = hP.c_block;
            if strcmpi(series.current, 'x')
                idx = series.x_num;
            elseif strcmpi(series.current, 'y')
                idx = series.y_num;
            end                
            
            if block == 0
                limit = hP.model.super.lim.SPE(idx);
            else
                limit = hP.model.lim{block}.SPE(idx);
            end
            
            % Which axis to add the limit to?
            set(ax, 'NextPlot', 'add')
            extent = axis;
            if strcmpi(series.current, 'x')
                hd = plot([limit, limit], [-1E10, 1E10], 'r-', 'linewidth', 1);
                xlabel(ax, 'Squared prediction error (SPE)')
            elseif strcmpi(series.current, 'y')
                hd = plot([-1E10, 1E10], [limit, limit], 'r-', 'linewidth', 1);                
                ylabel(ax, 'SPE')
            end
            set(hd, 'tag', 'hLimit', 'HandleVisibility', 'on');
            
            if series.x_num < 0 && series.y_num > 0
                title(ax, 'Squared prediction error (SPE)')
            end
            grid on            
            xlim(extent(1:2))
            ylim(extent(3:4)) 
            
            hPlot = findobj(ax, 'Tag', 'lvmplot_series');
            labels = hP.model.get_labels(hP.dim, hP.c_block);
            hP.label_scatterplot(hPlot, labels);
        end
        
        function [loadings_h, loadings_v, batchblock] = get_loadings_data(hP, series)
            % Correctly fetches the loadings data for the current block and dropdowns  \
            % Ugly hack to get batch blocks shown            
            block = hP.c_block;
            batchblock = [];
            loadings_h.data = [];
            
            % Single block data sets (PCA and PLS)
            if hP.model.B == 1
                if series.x_num > 0
                    loadings_h = hP.model.P{1}(:, series.x_num);
                    if isa(hP.model.blocks{1}, 'block_batch')
                        batchblock = hP.model.blocks{1};
                    end
                end
                loadings_v = hP.model.P{1}(:, series.y_num);
                if isa(hP.model.blocks{1}, 'block_batch')
                    batchblock = hP.model.blocks{1};
                end
                return
            end
            
            % Multiblock data sets: we also have superloadings
            if block == 0
                loadings_v = hP.model.super.P(:, series.y_num);
            else
                loadings_v = hP.model.P{block}(:, series.y_num);
                if isa(hP.model.blocks{block}, 'block_batch')
                    batchblock = hP.model.blocks{block};
                end
            end
                
            if series.x_num > 0
                if block == 0
                    loadings_h = hP.model.super.P(:, series.x_num); 
                else
                    loadings_h = hP.model.P{block}(:, series.x_num);
                    if isa(hP.model.blocks{block}, 'block_batch')
                        batchblock = hP.model.blocks{block};
                    end
                end
            end
        end        
        function loadings_plot(hP, series)            
            % Loadings plots for overall or per-block
            ax = hP.gca();
            [loadings_h, loadings_v, batchblock] = hP.model.get_loadings_data(hP, series);
            
            % Bar plot of the single loading
            if series.x_num <= 0
                % We need a bar plot for the loadings
                hPlot = findobj(ax, 'Tag', 'lvmplot_series');
                if hPlot
                    if ishandle(hPlot)
                        delete(hPlot)
                    end
                end
                hPlot = bar(ax, loadings_v, 'FaceColor', hP.opt.bar.facecolor);
                set(hPlot, 'Tag', 'lvmplot_series');
                
                
                % Batch plots are shown differently
                % TODO(KGD): figure a better way to deal with batch blocks
                if ~isempty(batchblock)
                    hP.annotate_batch_trajectory_plots(ax, hPlot, batchblock)
                end
                
                
                return
            end

            % Scatter plot of the loadings            
            hPlot = hP.set_data(ax, loadings_h, loadings_v);
            set(hPlot, 'LineStyle', 'none', 'Marker', '.', 'Color', [0, 0, 0])
        end
        function loadings_plot_annotate(hP, series)
            ax = hP.gca();
            hBar = findobj(ax, 'Tag', 'lvmplot_series');
            if isempty(hBar)
                return
            end
            if series.x_num > 0 && series.y_num > 0                         
                title(ax, 'Loadings plot')
                grid on
                xlabel(ax, ['p_', num2str(series.x_num)])
                ylabel(ax, ['p_', num2str(series.y_num)])
                hPlot = findobj(ax, 'Tag', 'lvmplot_series');
                if hP.model.B > 1
                    labels = hP.model.get_labels(hP.dim, hP.c_block);
                else
                    labels = hP.model.get_labels(hP.dim, 1);
                end
                hP.label_scatterplot(hPlot, labels);
            
            elseif series.y_num > 0                
                title(ax, 'Loading bar plot', 'FontSize', 15);
                grid on
                ylabel(ax, ['p_', num2str(series.y_num)])
                
                hBar = findobj(ax, 'Tag', 'lvmplot_series');
                if hP.model.B > 1
                    labels = hP.model.get_labels(hP.dim, hP.c_block);
                else
                    labels = hP.model.get_labels(hP.dim, 1);
                end
                hP.annotate_barplot(hBar, labels)
            end
        end
        
        function [VIP_data, batchblock] = get_VIP_data(hP, series)
            % Correctly fetches the VIP data for the current block and dropdowns            
            block = hP.c_block;
            batchblock = [];
            
            % Single block data sets (PCA and PLS)
            if hP.model.B == 1
                VIP_data = hP.model.stats{1}.VIP_a(:, series.y_num);
                if isa(hP.model.blocks{1}, 'block_batch')
                    batchblock = hP.model.blocks{1};
                end
                return
            end
            
            % Multiblock data sets: we also have superblock VIPs
            if block == 0
                VIP_data = hP.model.super.stats.VIP(:, series.y_num);
            else
                VIP_data = hP.model.stats{block}.VIP_a(:, series.y_num);
                if isa(hP.model.blocks{block}, 'block_batch')
                    batchblock = hP.model.blocks{block};
                end
            end
                
        end        
        function VIP_plot(hP, series)
            ax = hP.gca();
            [VIP_data, batchblock] = hP.model.get_VIP_data(hP, series);
            % Bar plot of the single VIP
            if series.x_num <= 0
                % We need a bar plot for the VIPs
                hPlot = findobj(ax, 'Tag', 'lvmplot_series');
                if hPlot
                    if ishandle(hPlot)
                        delete(hPlot)
                    end
                end
                hPlot = bar(ax, VIP_data, 'FaceColor', hP.opt.bar.facecolor);
                set(hPlot, 'Tag', 'lvmplot_series');
                
                set(ax, 'YLim', hP.get_good_limits(VIP_data, get(ax, 'YLim'), 'zero'))
                
                % Batch plots are shown differently
                if ~isempty(batchblock)
                    hP.annotate_batch_trajectory_plots(ax, hPlot, batchblock)
                end
            end
        end         
        function VIP_limits_annotate(hP, series)
            ax = hP.gca();
            hBar = findobj(ax, 'Tag', 'lvmplot_series');
            
            if series.x_num > 0 && series.y_num > 0                         
                title(ax, 'VIP plot')
                grid on
                xlabel(ax, ['VIP with A=', num2str(series.x_num)])
                ylabel(ax, ['VIP with A=', num2str(series.y_num)])
                
            elseif series.y_num > 0                
                title(ax, 'VIP bar plot')
                grid on
                ylabel(ax, ['VIP with A=', num2str(series.y_num)])
                
                if hP.model.B > 1
                    labels = hP.model.get_labels(hP.dim, hP.c_block);
                else
                    labels = hP.model.get_labels(hP.dim, 1);
                end
                if hP.c_block>0
                    if ~isa(hP.model.blocks{hP.c_block}, 'block_batch')
                        hP.annotate_barplot(hBar, labels)
                    end
                elseif hP.c_block==0
                    hP.annotate_barplot(hBar, labels)
                end
            end
        end

        % These should actually be in the PLS function
        function coeff_data = get_coefficient_data(hP, series)
            % Correctly fetches the coefficient data for the current block and dropdowns            
            
            % Single block data sets (PCA and PLS)
            if hP.model.B == 1
                subW = hP.model.W{1}(:, 1:series.y_num);
                subP = hP.model.P{1}(:, 1:series.y_num);
                subC = hP.model.super.C(:, 1:series.y_num);
                coeff_data = subW * inv(subP' * subW) * subC';
                return
            end
            
            block = hP.c_block;
            
            % Multiblock data sets: we also have superblock VIPs
            if block == 0
                coeff_data = [];                
            else
                subW = hP.model.W{block}(:, 1:series.y_num);
                subP = hP.model.P{block}(:, 1:series.y_num);
                subC = hP.model.super.C(:, 1:series.y_num);
                coeff_data = subW * inv(subP' * subW) * subC';
            end                
        end
        function coefficient_plot(hP, series)
            ax = hP.gca();
            % We cannot visualize coefficient from the "Overall block"
            if hP.hDropdown.getSelectedIndex == 0
                hP.hDropdown.setSelectedIndex(1)
                pause(1);
                return
            end
            
            
            
            coeff_data = hP.model.get_coefficient_data(hP, series);
            if isempty(coeff_data)
                hChild = get(ax, 'Children');
                delete(hChild)
                text(sum(get(ax, 'XLim'))/2, sum(get(ax, 'YLim'))/2, ...
                    'Coefficient plots not defined for the overall block', ...
                    'Color', [0.8 0.2 0.3], 'FontSize', 16, ...
                    'HorizontalAlignment', 'center')
                return
            end
            % Bar plot of the single coefficient
            if series.x_num <= 0
                % We need a bar plot for the coefficient plot
                hPlot = findobj(ax, 'Tag', 'lvmplot_series');
                if hPlot
                    if ishandle(hPlot)
                        delete(hPlot)
                    end
                end
                hPlot = bar(ax, coeff_data);%, 'FaceColor', hP.opt.bar.facecolor);
                set(hPlot, 'Tag', 'lvmplot_series');
                set(ax, 'YLim', hP.get_good_limits(coeff_data, get(ax, 'YLim'), 'zero'))
                
                % Batch plots are shown differently
                if hP.c_block>0 && isa(hP.model.blocks{hP.c_block}, 'block_batch')
                    if numel(hPlot)>1
                        hChild = get(ax, 'Children');
                        delete(hChild)
                        text(sum(get(ax, 'XLim'))/2, sum(get(ax, 'YLim'))/2, ...
                            'Coefficient plots not currently available for batch blocks.', ...
                            'Color', [0.8 0.2 0.3], 'FontSize', 16, ...
                            'HorizontalAlignment', 'center')
                        return
                    end
                    %hP.annotate_batch_trajectory_plots(ax, hPlot, hP.model.blocks{hP.c_block})
                end
            end
        end
        function coefficient_annotate(hP, series)
            ax = hP.gca();
            hBar = findobj(ax, 'Tag', 'lvmplot_series');
            if isempty(hBar)
                return
            end
            if series.x_num > 0 && series.y_num > 0                         
                title(ax, 'Coefficient plot')
                grid on
                xlabel(ax, ['Coefficient plot with A=', num2str(series.x_num)])
                ylabel(ax, ['Coefficient plot with A=', num2str(series.y_num)])
                
            elseif series.y_num > 0                
                title(ax, 'Coefficient bar plot')
                grid off
                ylabel(ax, ['Coefficient plot with A=', num2str(series.y_num)])
                
                if hP.model.B > 1
                    labels = hP.model.get_labels(hP.dim, hP.c_block);
                else
                    labels = hP.model.get_labels(hP.dim, 1);
                end
                
                breaks = get(hBar(1), 'XData');
                extent = axis;
                hold(ax, 'on')
                for n = 1:numel(breaks)-1
                    point = sum(breaks(n:n+1))/2.0;
                    plot([point point], extent(3:4), '-.', 'Color', [0.5 0.5 0.5])
                    
                end
                
                top = 0.95*extent(4);
                set(ax, 'XTickLabel', {})
                for n = 1:numel(breaks)
                     text(breaks(n), top, strtrim(labels{n}), 'Rotation', 0,...
                         'FontSize', 10, 'HorizontalAlignment', 'center');
                end
                set(ax, 'TickLength', [0 0], 'XLim', [breaks(1)-0.5 breaks(end)+0.5])


            end
        end
        
        function [R2_data, batchblock] = get_R2_per_variable_data(hP, series)
            % Correctly fetches the R2 data for the current block and dropdowns            
            block = hP.c_block;
            batchblock = [];
            
            % Single block data sets (PCA and PLS)
            if hP.model.B == 1
                R2_data = hP.model.stats{1}.R2Xk_a(:, 1:series.y_num);
                if isa(hP.model.blocks{1}, 'block_batch')
                    batchblock = hP.model.blocks{1};
                end
                return
            end
            
            % Multiblock data sets: we also have superblock VIPs
            if block == 0
                R2_data = zeros(hP.model.B,series.y_num);
                for b = 1:hP.model.B
                    R2_data(b,:) = hP.model.stats{b}.R2Xb_a;
                end
                % Wrong variable: this is the overal block R2X, per component
                %R2_data = hP.model.super.stats.R2X(:, 1:series.y_num);
            else
                R2_data = hP.model.stats{block}.R2Xk_a(:, 1:series.y_num);
                if isa(hP.model.blocks{block}, 'block_batch')
                    batchblock = hP.model.blocks{block};
                end
            end
                
        end 
        function R2_per_variable_plot(hP, series)
            ax = hP.gca();
            [R2_data, batchblock] = hP.model.get_R2_per_variable_data(hP, series);
            if series.x_num <= 0
                
                hPlot = findobj(ax, 'Tag', 'lvmplot_series');
                if hPlot
                    if ishandle(hPlot)
                        delete(hPlot)
                    end
                end
                
                if ~isempty(batchblock)
                    colour_order = {'r', [255, 102, 0]/255, [0.2, 0.8, 0.2], 'k', 'b', 'm'};
                    R2_data(isnan(R2_data)) = 0.0;
                    hPlot = zeros(size(R2_data, 2), 1);
                    for a = 1:size(R2_data, 2)
                        set(ax, 'Nextplot', 'add')
                        this_PC = reshape(sum(R2_data(:,1:a),2), batchblock.nTags, batchblock.J)';
                        colour = colour_order{mod(a,numel(colour_order))+1};
                        hPlot(a) = plot(this_PC(:), 'Color', colour);
                    end
                    tagNames = batchblock.labels{2};
                    
                    nSamples = batchblock.J;
                    x_r = xlim;
                    y_r = ylim;
                    xlim([x_r(1,1) nSamples*batchblock.nTags]);
                    tick = zeros(batchblock.nTags,1);
                    for k=1:batchblock.nTags
                        tick(k) = nSamples*k;
                    end
                    set(ax, 'LineWidth', 1);

                    for k=1:batchblock.nTags
                        % y was = diff(y_r)*0.9 + y_r(1),
                        text(round((k-1)*nSamples+round(nSamples/2)), ...
                            0.97, strtrim(tagNames(k,:)),... 
                            'FontWeight','bold','HorizontalAlignment','center');
                        %text(round((k-1)*nSamples+round(nSamples/2)), ...
                        %    diff(y_r)*0.05 + y_r(1), sprintf('%.1f',cum_area(k)), ...
                        %    'FontWeight','bold','HorizontalAlignment','center');
                    end

                    set(ax,'XTick',tick);
                    set(ax,'XTickLabel',[]);
                    set(ax,'Xgrid','On');
                    xlabel('Batch time repeated for each variable');
                    title('R2 per variable, per component');
                    set(hPlot, 'UserData', 'dont_annotate')
                else
                    hPlot = bar(ax, R2_data, 'stacked', 'FaceColor', [1,1,1]);
                    set(hPlot, 'Tag', 'lvmplot_series');
                    set(ax, 'YLim', [0.0, 1.0])
                end
                set(hPlot, 'Tag', 'lvmplot_series');
                
                
                
%                 % Batch plots are shown differently
%                 if ~isempty(batchblock)
%                     hP.annotate_batch_trajectory_plots(ax, hPlot, batchblock)
%                 end
            end
        end
        function R2_per_variable_plot_annotate(hP, series)
            ax = hP.gca();
            hBar = findobj(ax, 'Tag', 'lvmplot_series');
            if strcmpi(get(hBar, 'UserData'), 'dont_annotate')
                return
            end
            
            if series.x_num > 0 && series.y_num > 0                         
                title(ax, 'R2 plot')
                grid on
                xlabel(ax, ['R2 with A=', num2str(series.x_num)])
                ylabel(ax, ['R2 with A=', num2str(series.y_num)])
                
            elseif series.y_num > 0                
                title(ax, 'R2 bar plot')
                grid on
                ylabel(ax, ['R2 with A=', num2str(series.y_num)])
                
                if hP.model.B > 1
                    labels = hP.model.get_labels(hP.dim, hP.c_block);
                else
                    labels = hP.model.get_labels(hP.dim, 1);
                end
                if hP.c_block>0
                    if ~isa(hP.model.blocks{hP.c_block}, 'block_batch')
                        hP.annotate_barplot(hBar, labels, 'stacked')
                    end
                elseif hP.c_block==0
                    hP.annotate_barplot(hBar, labels, 'stacked')
                end
            end
        end

    end % end methods (static)
    
end % end classdef

%-------- Helper functions (usually used in 2 or more places). May NOT change ``self``
function ellipse_coords = ellipse_coordinates(scores, T2_limit_alpha)
% Calculates the ellipse coordinates for any two-column matrix of ``scores``
% at the given Hotelling's T2 ``T2_limit_alpha``.

 
% Standard equation of an ellipse for Hotelling's T2
%
%             x^2/a^2 + y^2 / b^2 = constant
%             (t_horiz/s_h)^2 + (t_vert/s_v)^2  =  T2_limit_alpha
%
% which is OK if (s_h)^2 and (s_v)^2 are from the covariance matrix of T, 
% i.e. they are the variances of scores being plotted, and T is orthogonal.
%
% For non-orthogonal T's we require a rotation.
%
% But how to calculate the rotation of the ellipse from the covariance matrix?
% Center the vectors to zero and do a PCA, specifically an eigenvector 
% decomposition, on the scores to find the rotation angle. Since PCA is an 
% orthogonal decompostation it will find the major axis and minor axis, which
% are required to be perpendicular to each other.
%
% Rotate the scores to align them them with the X-axis, calculate the ellipse
% for the rotated scores, then rotate the ellipse back.
% 
% Apply any shifting via mean centering.
% Also see: http://www.maa.org/joma/Volume8/Kalman/General.html

    h = 1;
    v = 2;

    % Calculate the centered score vectors:
    t_h_offset = mean(scores(:,h));
    t_v_offset = mean(scores(:,v));
    t_h = scores(:,h) - t_h_offset;
    t_v = scores(:,v) - t_v_offset;    
    scores = [t_h t_v];

    [vec,val]=eig(scores'*scores);
    val = diag(val);

    % Sort from smallest to largest
    [val, idx] = sort(val);
    vec = vec(:, idx);
    % Direction of the major axis of the ellipse, in direction of largest
    % eigenvalue/eigenvector pair:  i.e. we are just doing a PCA on the scores
    % to find the rotation direction.
    direction = vec(:,end);
    alpha = atan(direction(v) / direction(h));
    clockwise_rot_matrix = [cos(alpha) sin(alpha); -sin(alpha) cos(alpha)];
    scores_rot = (clockwise_rot_matrix * scores')';
    % NOTE: ``clockwise_rot_matrix`` is so similar to the eigenvectors! 
    %       it basically is just the rows flipped around.
    
    % Now construct a simple ellipse that is aligned with the axes (no rotation)
    s_h = std(scores_rot(:,1));
    s_v = std(scores_rot(:,2));
    alpha = 0;
    x_offset = 0;
    y_offset = 0;
    n_points = 100;

    %function [x, y] = ellipse_coordinates(s_h, s_v, T2_limit_alpha, n_points)

    %     We want the (t_horiz, t_vert) cooridinate pairs that form the T2 ellipse.
    %
    %     Inputs: s_h (std deviation of the score on the horizontal axis)
    %             s_v (std deviation of the score on the vertical axis)
    %             T2_limit_alpha: the T2_limit at the alpha confidence value
    %
    %     Equation of ellipse in canonical form (http://en.wikipedia.org/wiki/Ellipse)
    %
    %      (t_horiz/s_h)^2 + (t_vert/s_v)^2  =  T2_limit_alpha
    %
    %      s_horiz = stddev(T_horiz)
    %      s_vert  = stddev(T_vert)
    %      T2_limit_alpha = T2 confidence limit at a given alpha value (e.g. 99%)
    %
    %     Equation of ellipse, parametric form (http://en.wikipedia.org/wiki/Ellipse)
    %
    %     t_horiz = sqrt(T2_limit_alpha)*s_h*cos(t)
    %     t_vert  = sqrt(T2_limit_alpha)*s_v*sin(t)
    %
    %     where t ranges between 0 and 2*pi
    %
    %     Returns `n-points` equi-spaced points on the ellipse.


    % From Wikipedia:

    % An ellipse in general position can be expressed parametrically as the path 
    % of a point <math>(X(t),Y(t))</math>, where
    % X(t)=X_c + a\,\cos t\,\cos \varphi - b\,\sin t\,\sin\varphi
    % Y(t)=Y_c + a\,\cos t\,\sin \varphi + b\,\sin t\,\cos\varphi
    %as the parameter ''t'' varies from 0 to 2''?''.  
    % Here <math>(X_c,Y_c)</math> is the center of the ellipse, 
    %and <math>\varphi</math> is the angle between the <math>X</Math>-axis 
    % and the major axis of the ellipse.

    h_const = sqrt(T2_limit_alpha) * s_h;
    v_const = sqrt(T2_limit_alpha) * s_v;
    dt = 2*pi/(n_points-1);
    steps = (0:(n_points-1)) .* dt;
    cos_steps = cos(steps);
    sin_steps = sin(steps);
    x = x_offset + h_const.*cos_steps.*cos(alpha) - v_const.*sin_steps.*sin(alpha);
    y = y_offset + h_const.*cos_steps.*sin(alpha) + v_const.*sin_steps.*cos(alpha);
    
    counter_rotate = inv(clockwise_rot_matrix);
    ellipse_coords = (counter_rotate * [x(:) y(:)]')';
    
end

function basic_plot__scores(hP)
    % Show a basic set of score plots.

    % These are observation-based plots
    hP.dim = 1;
    
    if hP.model.A == 1
        hP.nRow = 1;
        hP.nCol = 1;
        hP.new_axis(1);
        hP.set_plot(1, {'Order', -1}, {'Scores', 1});
    elseif hP.model.A == 2
        hP.nRow = 1;
        hP.nCol = 1;
        hP.new_axis(1);
        hP.set_plot(1, {'scores', 1}, {'scores', 2});
    elseif hP.model.A >= 3
        % Show a t1-t2, a t2-t3, a t1-t3 and a Hotelling's T2 plot
        hP.nRow = 2;
        hP.nCol = 2;
        hP.new_axis([1, 2, 3, 4]);
        hP.set_plot(1, {'Scores', 1}, {'scores', 2});  % t1-t2
        hP.set_plot(2, {'Scores', 3}, {'scores', 2});  % t3-t2
        hP.set_plot(3, {'Scores', 1}, {'scores', 3});  % t1-t3
        hP.set_plot(4, {'Order', -1},  {'Hot T2', hP.model.A});  % Hot_T2 using all components
    end
end % ``basic_plot__scores``

function basic_plot__loadings(hP)
    % Show a basic set of loadings plots.

    % These are variable-based plots
    hP.dim = 2;
    
    if hP.model.A == 1
        hP.nRow = 1;
        hP.nCol = 1;
        hP.new_axis(1);
        hP.set_plot(1, {'Order', -1}, {'Loadings', 1});
    elseif hP.model.A == 2
        if hP.model.B == 1 && isa(hP.model.blocks{1}, 'block_batch')
            hP.nRow = 2;
            hP.nCol = 1;
            hP.new_axis([1 2]);
            hP.set_plot(1, {'Order', -1}, {'Loadings', 1});
            hP.set_plot(2, {'Order', -1}, {'Loadings', 2});
        else
            hP.nRow = 1;
            hP.nCol = 1;
            hP.new_axis(1);
            hP.set_plot(1, {'Loadings', 1}, {'Loadings', 2});
        end
    elseif hP.model.A >= 3
        if hP.model.B == 1 && isa(hP.model.blocks{1}, 'block_batch')
            hP.nRow = 2;
            hP.nCol = 1;
            hP.new_axis([1 2]);
            hP.set_plot(1, {'Order', -1}, {'Loadings', 1});
            hP.set_plot(2, {'Order', -1}, {'Loadings', 2});
        else
            % Show a t1-t2, a t2-t3, a t1-t3 and a Hotelling's T2 plot
            hP.nRow = 2;
            hP.nCol = 2;
            hP.new_axis([1, 2, 3, 4]);
            hP.set_plot(1, {'Loadings', 1}, {'Loadings', 2});
            hP.set_plot(2, {'Loadings', 3}, {'Loadings', 2});
            hP.set_plot(3, {'Loadings', 1}, {'Loadings', 3}); 
            hP.set_plot(4, {'Order', -1},  {'VIP', hP.model.A});  % VIP using all components
        end
    end
end % ``basic_plot__loadings``

function basic_plot__weights(hP)
    % Show a basic set of loadings plots.

    % These are variable-based plots
    hP.dim = 2;
    
    if hP.model.A == 1
        hP.nRow = 1;
        hP.nCol = 1;
        hP.new_axis(1);
        hP.set_plot(1, {'Order', -1}, {'Weights', 1});
    elseif hP.model.A == 2
        if hP.model.B == 1 && isa(hP.model.blocks{1}, 'block_batch')
            hP.nRow = 2;
            hP.nCol = 1;
            hP.new_axis([1 2]);
            hP.set_plot(1, {'Order', -1}, {'Weights', 1});
            hP.set_plot(2, {'Order', -1}, {'Weights', 2});
        else
            hP.nRow = 1;
            hP.nCol = 1;
            hP.new_axis(1);
            hP.set_plot(1, {'Weights', 1}, {'Weights', 2});
        end
    elseif hP.model.A >= 3
        if hP.model.B == 1 && isa(hP.model.blocks{1}, 'block_batch')
            hP.nRow = 2;
            hP.nCol = 1;
            hP.new_axis([1 2]);
            hP.set_plot(1, {'Order', -1}, {'Weights', 1});
            hP.set_plot(2, {'Order', -1}, {'Weights', 2});
        else
            % Show a t1-t2, a t2-t3, a t1-t3 and a Hotelling's T2 plot
            hP.nRow = 2;
            hP.nCol = 2;
            hP.new_axis([1, 2, 3, 4]);
            hP.set_plot(1, {'Weights', 1}, {'Weights', 2});
            hP.set_plot(2, {'Weights', 3}, {'Weights', 2}); 
            hP.set_plot(3, {'Weights', 1}, {'Weights', 3}); 
            hP.set_plot(4, {'Order', -1},  {'VIP', hP.model.A});  
        end
    end
end % ``basic_plot__weights``

function batch_plot__weights(hP)
    % Show weights plots for batch systems

    % These are variable-based plots
    hP.dim = 2;
    
    if hP.model.A == 1
        hP.nRow = 1;
        hP.nCol = 1;
        hP.new_axis(1);
        hP.set_plot(1, {'Order', -1}, {'Weights', 1});
    elseif hP.model.A >= 2
        hP.nRow = 2;
        hP.nCol = 1;
        hP.new_axis([1 2]);
        hP.set_plot(1, {'Order', -1}, {'Weights', 1});
        hP.set_plot(2, {'Order', -1}, {'Weights', 2});
    end
end % ``batch_plot__weights``

function basic_plot__spe(hP)
    % These are observation-based plots
    hP.dim = 1;
    
    % TODO(KGD): add the ability to plot data from multiple blocks
    % i.e. click plot, then change dropdown: will only change plot in that
    % subplot
    hP.nRow = 1;
    hP.nCol = 1;
    hP.new_axis(1);
    hP.set_plot(1, {'Order', -1}, {'SPE', hP.model.A});
end % ``basic_plot__spe``

function basic_plot__predictions(hP)
    % These are observation-based plots
    hP.dim = 1;
    
    % TODO(KGD): add the ability to plot data from multiple blocks
    % i.e. click plot, then change dropdown: will only change plot in that
    % subplot
    M = min(hP.model.M, 6);
    [hP.nRow hP.nCol] = hP.subplot_layout(M);
    hP.new_axis(1:M);
    for m = 1:M
        hP.set_plot(m, {'Observations', m}, {'Predictions', m});
    end
end % ``basic_plot__predictions``

function basic_plot__VIP(hP)
    % These are variable-based plots
    hP.nRow = 1;
    hP.nCol = 1;
    hP.dim = 2;
    hP.new_axis(1);
    hP.set_plot(1, {'Order', -1}, {'VIP', hP.model.A});
end % ``basic_plot__VIP``

function basic_plot__coefficient(hP)
    % These are variable-based plots
    hP.nRow = 1;
    hP.nCol = 1;
    hP.dim = 2;
    hP.new_axis(1);
    hP.set_plot(1, {'Order', -1}, {'Coefficient', hP.model.A});
end % ``basic_plot__VIP``

function basic_plot_R2_variable(hP)
    % These are variable-based plots
    hP.nRow = 1;
    hP.nCol = 1;
    hP.dim = 2;
    hP.new_axis(1);
    hP.set_plot(1, {'Order', -1}, {'R2 (per variable)', hP.model.A});
end % ``basic_plot_R2_variable``

function basic_plot_R2_Y_variable(hP)
    % These are variable-based plots
    hP.nRow = 1;
    hP.nCol = 1;
    hP.dim = 2;
    hP.new_axis(1);
    hP.set_plot(1, {'Order', -1}, {'R2-Y-variable', hP.model.A});
end % ``basic_plot_R2_Y_variable``

function basic_plot_R2_component(hP)
    % These are model (component)-based plots
    hP.nRow = 1;
    hP.nCol = 1;
    hP.dim = 0;
    hP.new_axis(1);
    hP.set_plot(1, {'Order', -1}, {'R2 (per component)', hP.model.A});
end % ``basic_plot_R2_variable``