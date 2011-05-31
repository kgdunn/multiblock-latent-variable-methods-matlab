% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% Distributed under the BSD license.
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
    
    events
        % Model building events
        % ---------------------
        build_launch
        
        build_initializestorage_launch
        build_initializestorage_action
        build_initializestorage_finish

        build_preprocess_launch
        build_preprocess_core
        build_preprocess_action
        build_preprocess_finish
        
        build_merge_launch
        build_merge_action
        build_merge_finish
        
        build_calculate_launch
        
        % Collect all criteria to decide if loop to add next component should
        % be run. All criteria must be true if the next component is to be
        % added
        build_calculate_ifaddnext__loop_entry_condition
        
        build_calculate_loop_addcomponent_action
        build_calculate_loop_deflate_action
        build_calculate_loop_calcstats_action
        build_calculate_loop_calclimits_action
        
        % Sometimes we can only decide on a component after it is calculated.
        % This 
        
        build_calculate_ifaddnext__loop_exit_condition
        
        build_calculate_finish
                
        build_finish
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
            self.opt = self.options('default');

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
        
        function terminate_adding = randomization_should_terminate(self)
            % Determines whether a new component should be added using a
            % randomization test.
            %
            % Wiklund, et al., J Chemometrics, 21, p427-439, 2007, "A randomization
            % test for PLS component selection". http://dx.doi.org/10.1002/cem.1086
            %
            % Also see: http://www.springerlink.com/content/u2u43772t603k7w7/
            %
            % Based on results in that paper we will keep fitting components until
            % the deemed risk is too high.  The risk is quantified by a the number of
            % risk points.  After 2 points we stop adding components and revert to the
            % component where we had 1.0 points, or fewer.
            %
            % Points are cumulative. If we have 1.0 points, and we add a new
            % component, then the points from previous components are still retained.

            max_A = min(self.N, self.K);
            
            if self.A == 0
                % This is the very first iteration (no components fitted yet)
                % so we should try fitting a single component.
                terminate_adding = false;
                return
            end

            % Return if we've reached the maximum number of components supported by
            % this model.
            % Return if the user has clicked the "Stop adding" button during
            % randomization risk assessment.
            if self.A >= max_A || self.opt.randomize_test.risk_statistics{self.A}.stop_early
                self.opt.randomize_test.last_worthwhile_A = self.A;
                terminate_adding = true;
                return
            end

            % We've already evaluated the risk-free cases, now let's evaulate the risky ones.
            %
            % 1.	Let risk = \frac{\text{number of}\,\,S_g\,\,\text{values exceeding}\,\,S_0}{G}
            %
            %     *	If risk :math:`\geq 0.08`, then ``points = points + 2``, as there is a high risk, one in 12 chance, we are accepting a component that should not be accepted.
            %
            %     *	or, if :math:`0.03 < \text{risk} < 0.08` then ``points = points + 1``  (moderately risky to accept this component)
            %
            %     *	finally, if :math:`risk \leq 0.03` then we accept the component without accumulating any points, however, we might still add some points if the correlation, :math:`S_0` is small (see next step).
            %
            % 2.	Note that :math:`S_0` represents the correlation between :math:`t_a` and the :math:`u_a`, which is nothing more than a scaled version of the objective function of the PLS model, which each component is trying to maximize, subject to certain constraints.  We accumulate risk based on the strength of this correlation as follows:
            %
            %     *	If :math:`S_0 \geq 0.50`, then we do not augment our risk, as this is a strong correlation
            %
            %     *	Or, if :math:`0.35 < S_0 < 0.50`, then ``points = points + 0.5`` (weak correlation between :math:`t_a` and :math:`u_a`)
            %
            %     *	Or, if :math:`S_0 \leq 0.35` then ``points = points + 1.0`` (very weak correlation between :math:`t_a` and :math:`u_a`)
            %
            % We stop adding components when the total risk points *accumulated on the current and all previous components* equals or exceeds 2.0.  We revert to the component where we had a risk points of 1.0 or less and stop adding components.

            randstats = self.opt.randomize_test.risk_statistics{self.A};
            current_risk = randstats.num_G_exceeded  / randstats.nperm;
            correlation = self.opt.randomize_test.test_statistic(self.A);
            current_points = self.opt.randomize_test.points;
            % Assess risk based on the number of violations in randomization
            if current_risk >= 0.10
                self.opt.randomize_test.points = self.opt.randomize_test.points + 2;
            elseif current_risk >= 0.05
                self.opt.randomize_test.points = self.opt.randomize_test.points + 1;
            elseif current_risk >= 0.01
                self.opt.randomize_test.points = self.opt.randomize_test.points + 0;
            elseif current_risk < 0.01
                self.opt.randomize_test.last_worthwhile_A = self.A;
                %self.opt.randomize_test.points = max(0, self.opt.randomize_test.points - 1.0);
                self.opt.randomize_test.risk_statistics{self.A}.points = self.opt.randomize_test.points - current_points;
                terminate_adding = false;
                return
            end

            % Assess risk also based on the strength of the t_a vs u_a correlation
            if correlation >= 0.5
                self.opt.randomize_test.points = self.opt.randomize_test.points + 0;
            elseif correlation >= 0.35
                self.opt.randomize_test.points = self.opt.randomize_test.points + 0.5;
            else
                self.opt.randomize_test.points = self.opt.randomize_test.points + 1.0;
            end

            self.opt.randomize_test.risk_statistics{self.A}.points = self.opt.randomize_test.points - current_points;

            if self.opt.randomize_test.points >= 2.0
                terminate_adding = true;
            else
                terminate_adding = false;
                if self.opt.randomize_test.points <= 1.0
                    self.opt.randomize_test.last_worthwhile_A = self.A;
                end
            end           
            
        end % ``randomization_add_next``
        
        function self = randomization_test(self, current_A) 
       
            nperm = self.opt.randomize_test.permutations;
            self.opt.randomize_test.test_statistic = self.zeroexp([1, current_A], ...
                                     self.opt.randomize_test.test_statistic);
            
            
            % TODO(KGD): ensure there is variation left in the X and Y blocks
            % to support extracting components.  Right now that error check is 
            % left out of the loop - for speed. 
 
            self.opt.randomize_test.test_statistic(current_A) = ...
                                      self.randomization_objective(current_A);
 
            self.randomization_test_launch();
            
            
            capture_more_permutations = true;
            rounds = 0;
            stats = struct; %#ok<*PROP>
            stats.mean_G = 0;
            stats.std_G = 0;
            stats.num_G_exceeded = 0;
            stats.nperm = 0;
            stats.stop_early = false;
            
            if self.opt.randomize_test.show_progress
                self.progressbar(sprintf('Risk of component %d', current_A));
            end
            while capture_more_permutations && ...
                                 (rounds < self.opt.randomize_test.max_rounds)
                % Run the purmutation tests at least once, in a group of G,
                % but maybe more, especially when the risk is borderline:                
                permuted_stats = zeros(nperm, 1);
                
                % Use the ``fit_PLS`` function internally, with a special flag to
                % return early
                previous_tolerance = self.opt.tolerance;
                self.opt.tolerance = 1e-2;
                itern = zeros(nperm,1);
                
                for g = 1:nperm
                    perc = floor(g/nperm*100);
                    num_G = stats.num_G_exceeded + sum(permuted_stats > ...
                           self.opt.randomize_test.test_statistic(current_A));
                    den_G = stats.nperm + g;
                    if self.opt.randomize_test.show_progress
                        stats.stop_early = 0;
                        self.progressbar(perc/100)
                        self.progressbar(sprintf('Risk of component %d. Risk so far=%d out of %d models.', ...
                                              current_A, num_G, den_G));
                        if stats.stop_early
                            nperm = g-1;
                            permuted_stats = permuted_stats(1:g-1);
                            rounds = Inf;  % forces it to exit
                            break;
                        end
                    end
                    
                    % Set the random seed: to ensure we can reproduce calculations.
                    rand('twister', g+rounds*nperm); %#ok<RAND>
                    out = self.randomization_permute_and_build();
                    
                    % Next, calculate the statistic under test and store it
                    permuted_stats(g) = self.randomization_objective(current_A, out);
                    itern(g) = out.itern;
                    
                end
                self.opt.tolerance = previous_tolerance;
                
                num_G = sum(permuted_stats > self.opt.randomize_test.test_statistic(current_A));
                
                % TODO(KGD): put this into the above function
                prev_ssq = stats.std_G^2 * (stats.nperm-1) + ...
                                                   stats.nperm*stats.mean_G^2;
                curr_ssq = self.ssq(permuted_stats);
                stats.mean_G = (stats.mean_G*stats.nperm + ...
                   block_base.nanmean(permuted_stats)*nperm)/(stats.nperm+nperm);
                
                % Guard against getting a negative under the square root
                % This approach quickly becomes inaccurate when using many rounds.
                stats.std_G = sqrt(((prev_ssq + curr_ssq) - ...
                    (stats.nperm+nperm)*stats.mean_G^2)/(stats.nperm+nperm-1));
                stats.num_G_exceeded = stats.num_G_exceeded + num_G;
                stats.nperm = stats.nperm + nperm;
                rounds = rounds + 1;                
                
                % Assume we've got enough randomized values to make an accurate risk
                % assessment.
                capture_more_permutations = false;
                risk = stats.num_G_exceeded / stats.nperm * 100;
                bounds = self.opt.randomize_test.risk_uncertainty;
                if any( (risk > bounds(1)) && risk < bounds(2) )
                    % Do another round of permutations to clarify risk level.
                    capture_more_permutations = true;
                end
            end
            if self.opt.randomize_test.show_progress
                self.progressbar(1.0)
            end
            self.opt.randomize_test.risk_statistics{current_A} = stats;
            
            % Set the Y-block back to its usual order
            self.randomization_test_finish();
            
        end % ``randomization_test``
        
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
            
            
            if self.opt.show_progress
                self.progressbar(sprintf('Calculating monitoring limits for model'));
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
                if self.opt.show_progress
                    perc = floor(n/self.N*100);
                    self.progressbar(perc/100) %sprintf('Processing batches for limits %d. [%d%%]',n, perc));
                    %if stop_early                             
                    %    break; 
                    %end	
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
                            SPE_j_temp(n,j) = self.ssq(out.newb{b}.data(1, idx_beg:idx_end));
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
                        %self.stats{b}.SPE_j = self.zeroexp([dblock.N, dblock.J], self.stats{b}.SPE_j, true);
                        
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
            
            % Randomization-based risk calculation
            self.model.stats.risk = struct;
            self.model.stats.risk.rate = [];
            self.model.stats.risk.objective = [];
            self.model.stats.risk.stats = {};
            
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
            
            self.model.stats.timing = self.zeroexp([A, 1], self.model.stats.timing);
            self.model.stats.itern = self.zeroexp([A, 1], self.model.stats.itern);
            self.model.stats.risk.rate = self.zeroexp([A, 1], self.model.stats.risk.rate);
            self.model.stats.risk.objective = self.zeroexp([A, 1], self.model.stats.risk.objective);
            
            
            % Storage for each block
            for b = 1:self.B
                dblock = self.blocks{b};
                self.P{b} = self.zeroexp([dblock.K, A], self.P{b});  % block loadings; 
                self.T{b} = self.zeroexp([dblock.N, A], self.T{b});  % block scores
                self.W{b} = self.zeroexp([dblock.K, A], self.W{b});  % PLS block weights
               %self.R{b} = self.zeroexp([dblock.K, A], self.R{b});  % PLS block weights
                self.S{b} = self.zeroexp([A, A], self.S{b});         % score scaling factors
                
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
                self.stats{b}.SPE = self.zeroexp([dblock.N, A], self.stats{b}.SPE);
                
                
                % Instantaneous SPE limit using all A components (batch models)
                % N x J
                %self.stats{b}.SPE_j = self.zeroexp([dblock.N, dblock.J], self.stats{b}.SPE_j, true);

                
                % Baseline value for all R2 calculations: before any components are
                % extracted, but after the data have been preprocessed.
                % 1 x K(b)
                self.stats{b}.start_SS_col = self.zeroexp([1, dblock.K], self.stats{b}.start_SS_col);
                
                % R^2 for every variable in the block, per component (not cumulative)
                % K(b) x A
                self.stats{b}.R2Xk_a = self.zeroexp([dblock.K, A], self.stats{b}.R2Xk_a);
                
                % R^2 for every variable in the Y-block, per component (not cumulative)
                % M x A
                self.super.stats.R2Yk_a = self.zeroexp([self.M, A], self.super.stats.R2Yk_a);
                
                % Sum of squares for each column in the block, prior to the component
                % being extracted.
                % K(b) x A
                self.stats{b}.col_ssq_prior = self.zeroexp([dblock.K, A], self.stats{b}.col_ssq_prior);                
                                
                % R^2 for the block, per component
                % 1 x A
                self.stats{b}.R2Xb_a = self.zeroexp([1, A], self.stats{b}.R2Xb_a);
                
                % Sum of squares explained for this component
                % 1 x A
                self.stats{b}.SSQ_exp = self.zeroexp([1, A], self.stats{b}.SSQ_exp);

                % VIP value using all 1:A components (only last column is useful though)
                % K(b) x A
                self.stats{b}.VIP_a = self.zeroexp([dblock.K, A], self.stats{b}.VIP_a);
                
                % Overall T2 value for each observation in the block using
                % all components 1:A
                % N x A
                self.stats{b}.T2 = self.zeroexp([dblock.N, A], self.stats{b}.T2);
                
                % Instantaneous T2 limit using all A components (batch models)
                %self.stats{b}.T2_j = self.zeroexp([dblock.N, dblock.J], self.stats{b}.T2_j);

                % Modelling power = 1 - (RSD_k)/(RSD_0k)
                % RSD_k = residual standard deviation of variable k after A PC's
                % RSD_0k = same, but before any latent variables are extracted
                % RSD_0k = 1.0 if the data have been autoscaled.
                %self.stats{b}.model_power = self.zeroexp([1, dblock.K], self.stats{b}.model_power);

                % Actual limits for the block: to be calculated later on
                % ---------------------------
                % Limits for the (possibly time-varying) scores
                % 1 x A
                % NOTE: these limits really only make sense for uncorrelated
                % scores (I don't think it's too helpful to monitor based on limits 
                % from correlated variables)
                self.lim{b}.t = self.zeroexp([1, A], self.lim{b}.t);
                %self.lim{b}.t_j = self.zeroexp([dblock.J, A], self.lim{b}.t_j, true); 

                % Hotelling's T2 limits using A components (column)
                % (this is actually the instantaneous T2 limit,
                % but we don't call it that, because at time=J the T2 limit is the
                % same as the overall T2 limit - not so for SPE!).
                % 1 x A
                self.lim{b}.T2 = self.zeroexp([1, A], self.lim{b}.T2);            

                % Overall SPE limit for the block using 1:A components (use the 
                % last column for the limit with all A components)
                % 1 x A
                self.lim{b}.SPE = self.zeroexp([1, A], self.lim{b}.SPE);

                % SPE instantaneous limits using all A components
                %self.lim{b}.SPE_j = self.zeroexp([dblock.J, 1], self.lim{b}.SPE_j);

            end
            
            % Superblock storage
            % ------------------
            
            % Summary scores from each block: these match self.T{b}, so we
            % don't really need to store them in the superblock structure.
            self.super.T_summary = self.zeroexp([self.N, self.B, A], self.super.T_summary);
            self.super.T = self.zeroexp([self.N, A], self.super.T);
            self.super.P = self.zeroexp([self.B, A], self.super.P);
            self.super.W = self.zeroexp([self.B, A], self.super.W);
            self.super.C = self.zeroexp([self.M, A], self.super.C);  % PLS Y-space loadings
            self.super.U = self.zeroexp([self.N, A], self.super.U);  % PLS Y-space scores
               
            
            % T2, using all components 1:A, in the superblock
            self.super.T2 = self.zeroexp([self.N, A], self.super.T2);
            
            % Limits for superscore entities
            self.super.lim.t = self.zeroexp([1, A], self.super.lim.t);
            self.super.lim.T2 = self.zeroexp([1, A], self.super.lim.T2);            
            self.super.lim.SPE = self.zeroexp([1, A], self.super.lim.SPE);
            
            % Statistics in the superblock
            self.super.stats.R2X = self.zeroexp([1, A], self.super.stats.R2X);
            self.super.stats.R2Y = self.zeroexp([1, A], self.super.stats.R2Y);
            self.super.stats.ssq_Y_before = [];
            
            self.super.stats.SSQ_exp = self.zeroexp([1, A], self.super.stats.SSQ_exp);
            % The VIP's for each block, and the VIP calculation factor
            self.super.stats.VIP = self.zeroexp([self.B, 1], self.super.stats.VIP);
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
                    if ~isempty(self.blocks{b}.labels)
                        if ~isempty(self.blocks{b}.labels{dim})
                            out = self.blocks{b}.labels{dim};
                            return
                        end
                    end
                end
            else
                if req_block ~= 0  
                    if ~isempty(self.blocks{req_block}.labels)
                        if ~isempty(self.blocks{req_block}.labels{dim})
                            out = self.blocks{req_block}.labels{dim};
                            return
                        end
                    end
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
        randomization_objective(self) % the randomization test's objective function
        randomization_test_launch(self) % Setup required before randomization
        randomization_test_finish(self) % Cleanup required after randomization
        randomization_permute_and_build(self) % Permutes data and build model
    end % end: methods (abstract)
    
    % Subclasses may not redefine these methods
    methods (Sealed=true, Static=true)
        
        function opt = options(varargin)
            % Returns options for all latent variable model types.

            opt.md_method = 'scp';
            opt.max_iter = 15000;
            opt.show_progress = true;         % Show a progress bar with option to cancel
            opt.min_lv = -1;
            opt.max_lv = inf;
            opt.build_now = true;
            opt.tolerance = sqrt(eps);
            opt.stop_now = false;             % Periodically checked; used to stop any loops

            opt.mbpls.block_scale_X = true;                  % The various X-blocks should be block scaled
            opt.mbpls.deflate_X = true;                      % No need to deflate the X-blocks, but, we must account for it in the PLS model


            opt.batch.calculate_monitoring_limits = false;   % Calculate monitoring limits for batch blocks
            opt.batch.monitoring_level = 0.95;
            opt.batch.monitoring_limits_show_progress = true;

            opt.randomize_test = struct;
            opt.randomize_test.use = false;
            opt.randomize_test.points = 0;                    % Start with zero points
            opt.randomize_test.risk_uncertainty = [0.5 10.0]; % Between these levels the percentage risk is considered uncertain, and could be due 
                                                             % to the  randomization values.  So will do more permutationsm, to a maximum of 3 
                                                             % times the default amount, to more clearly define the risk level.
            opt.randomize_test.permutations = 500;           % Default number of permutations.
            opt.randomize_test.max_rounds = 20;              % We will do at most 1000 permutations to assess risk
            opt.randomize_test.test_statistic = [];
            opt.randomize_test.risk_statistics = cell(1,1);
            opt.randomize_test.last_worthwhile_A = 0;        % The last worthwhile component added to the model
            opt.randomize_test.show_progress = true;
            opt.randomize_test.temp_data = {};               % Temporary data during the randomization routine

            % Cross-validation is not working as intended.
            % 
            % opt.cross_val = struct;
            % opt.cross_val.use = false;  % use cross-validation or not
            % opt.cross_val.groups = 5;   % not of groups to use
            % opt.cross_val.start_at = 4; % sequential counting of groups starts here
            % opt.cross_val.strikes = 0;  % number of strikes encountered already
            % opt.cross_val.PRESS = [];
            % opt.cross_val.PRESS_0 = [];
            
        end % ``options``
        
        function [T2, varargout] = mahalanobis_distance(T, varargin)
            % If given ``S``, an estimate of the variance-covariance matrix,
            % then it will use that, instead of an estimated var-cov matrix.
            
            % TODO(KGD): calculate this in a smarter way. Can create unnecessarily
            % large matrices
            n = size(T, 1);
            if nargin == 2
                estimated_S = varargin{1};
            else
                covariance = (T'*T)/n;
                estimated_S = inv(covariance);
            end
            
            % Equivalent of T2 = diag(T * estimated_S * T'); but more 
            % memory efficient
            T2 = zeros(n,1);            
            for idx = 1:n
                T2(idx) = T(idx,:) * estimated_S * T(idx,:)';
            end
            if nargout > 1
                varargout{1} = estimated_S;
            end            
        end
        
        function q = chi2inv(p, v)
             %CHISQQ	Quantiles of the chi-square distribution.
            %	Q = CHISQQ(P,V) satisfies P(X < Q) = P, where X follows a
            %	chi-squared distribution on V degrees of freedom.
            %	V must be a scalar.
            %
            %	See also CHISQP.
            %
            %	Gordon K Smyth, University of Queensland, gks@maths.uq.edu.au
            %	27 July 1999
            %
            %	Reference:  Johnson and Kotz (1970). Continuous Univariate
            %	Distributions, Volume I. Wiley, New York.
            %
            %   From: http://www.statsci.org/matlab/contents.html
            %   Permission was granted to use code; see email on 17 May 2011.
            %
            % Another alternative might be: 
            % http://www.spatial-econometrics.com/
            q = 2*mblvm.gammaq(p,v/2);
        end
        
        function p = gammap(q,a)
            %GAMMAP Gamma distribution function.
            %	GAMMAP(Q,A) is Pr(X < Q) where X is a Gamma random variable with
            %	shape parameter A.  A must be scalar.
            %
            %	See also GAMMAQ, GAMMAR.
            %
            %   GKS  2 August 1998.
            %
            %   From: http://www.statsci.org/matlab/contents.html
            %   Permission was granted to use code; see email on 17 May 2011.

            if a < 0, error('Gamma parameter A must be positive'); end
            p = gammainc(q,a);
            k = find(q < 0);
            if any(k), p(k) = zeros(size(k)); end
        end
        
        function q = gammaq(p,a)
            %GAMMAQ	Gamma distribution quantiles.
            %	Q = GAMMAQ(P,A) satisfies Pr(X < Q) = P where X follows a
            %	Gamma distribution with shape parameter A > 0.
            %	A must be scalar.
            %
            %	See also GAMMAP, GAMMAR

            %	Gordon Smyth, gks@maths.uq.edu.au, University of Queensland
            %	2 August 1998

            %	Method:  Newton iteration on the scale of log(X), starting
            %	from point of inflexion of cdf.  Monotonic convergence is
            %	guaranteed.
            %
            %   From: http://www.statsci.org/matlab/contents.html
            %   Permission was granted to use code; see email on 17 May 2011.

            if any(p < 0 | p > 1), error('P must be between 0 and 1'); end

            if a == 0.5, q = 0.5*normq((1-p)/2).^2; return; end
            if a == 1, q = -log(1-p); return; end

            G = gammaln(a);
            x = log(a);
            q = a;
            for i=1:10,
                x = x - (mblvm.gammap(q,a) - p) ./ exp(a*x - q - G);
                q = exp(x);
            end

            k = find((p>0 & p<1.5e-4) | (p>0.99999 & p<1));
            if any(k),
               X = x(k);
               Q = q(k);
               P = p(k);
               for i=1:10,
                  X = X - (mblvm.gammap(Q,a) - P) ./ exp(a*X - Q - G);
                  Q = exp(X);
               end;
               q(k) = Q;
            end

            k = find(p == 0);
            if any(k), q(k) = zeros(size(k)); end

            k = find(p == Inf);
            if any(k), q(k) = ones(size(k)); end
        end
        
        function q = finv(p,v1,v2)
            %FQ	F distribution quantiles.
            %	Q = FQ(P,V1,V2) satisfies Pr(X < Q) = P, where X follows
            %	and F distribution on V1 and V2 degrees of freedom.
            %	V1 and V2 must be scalars.
            %
            %	See also FP
            %
            %	Gordon Smyth, gks@maths.uq.edu.au, University of Queensland
            %
            %   From: http://www.statsci.org/matlab/contents.html
            %   Permission was granted to use code; see email on 17 May 2011.

            q = mblvm.betainv(p,v1/2,v2/2);
            q = v2/v1 * q./(1-q);
        end
        
        function q = betainv(p,a,b)
            %BETAQ	Beta distribution quantiles.
            %	Q = BETAQ(P,A,B) satisfies Pr(X < Q) = P where X follows a
            %	BETA distribution with parameters A > 0 and B > 0.
            %	A and B must be scalars.
            %
            %	See also BETAP

            %	Gordon Smyth, smyth@wehi.edu.au,
            %	Walter and Eliza Hall Institute of Medical Research
            %	1 August 1998

            %	Method:  Work with the distribution function of log(X/(1-X)).
            %	The cdf of this distribution has one point of inflexion at the
            %	the mode of the distribution at log(A/B).  Newton's method
            %	converges monotonically from this starting value.
            %
            %   From: http://www.statsci.org/matlab/contents.html
            %   Permission was granted to use code; see email on 17 May 2011.

            q = zeros(size(p));

            k = find(p >= 1);
            if any(k), q(k) = ones(size(k)); end

            k = find(p > 0 & p < 1);
            if any(k),
               P = p(k);
               B = betaln(a,b);
               r = a/(a+b);
               x = log(a/b);
               ex = a/b;
               for i=1:6,
                  x = x - (mblvm.betap(r,a,b) - P) ./ exp(a*x - (a+b)*log(1+ex) - B);
                  ex = exp(x);
                  r = ex./(1+ex);
               end
               q(k) = r;
            end
        end
        
        function p = betap(x,a,b)
            %BETAP	Beta cumulative distribution function.
            %	BETAP(X,A,B) gives P(X < x) where X follows a BETA distribution
            %	with parameters A > 0 and B > 0.  A and B must be scalars.
            %
            %	See also BETAQ, BETAR

            %	Gordon Smyth, gks@maths.uq.edu.au, University of Queensland
            %	31 July 1998
            %
            %   From: http://www.statsci.org/matlab/contents.html
            %   Permission was granted to use code; see email on 17 May 2011.

            p = zeros(size(x));

            k = find(x >= 1);
            if any(k), p(k) = ones(size(k)); end

            k = find(x > 0 & x < 1);
            if any(k), p(k) = betainc(x(k),a,b); end
        end
        
        function q = tinv(p,v)
            %TQ	t distribution quantiles.
            %	Q = TQ(P,V) satisfies Pr(T < Q) = P where T follows a
            %	t-distribution on V degrees of freedom.
            %	V must be a scalar.
            %
            %	Gordon Smyth, University of Queensland, gks@maths.uq.edu.au
            %	2 August 1998
            % 
            %   From: http://www.statsci.org/matlab/contents.html
            %   Permission was granted to use code; see email on 17 May 2011.
            
            if v <= 0, error('Degrees of freedom must be positive.'); end;

            q = sign(p-0.5).*sqrt( mblvm.finv(2.*abs(p-0.5),1,v) );
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

        function  v = robust_scale(a)
        % Using the formula from Mosteller and Tukey, Data Analysis and Regression,
        % p 207-208, 1977.
            n = numel(a);
            location = median(a);
            spread_MAD = median(abs(a-location));
            ui = (a - location)/(6*spread_MAD);    
            % Valid u_i values used in the summation:
            vu = ui.^2 <= 1;
            num = (a(vu)-location).^2 .* (1-ui(vu).^2).^4;
            den = (1-ui(vu).^2) .* (1-5*ui(vu).^2);
            v = n * sum(num) / (sum(den))^2;
        end
        
        function progressbar(varargin)
            
            % Used under the BSD license
            % http://www.mathworks.com/matlabcentral/fileexchange/6922-progressbar
            %
            % Description:
            %   progressbar() provides an indication of the progress of some task using
            % graphics and text. Calling progressbar repeatedly will update the figure and
            % automatically estimate the amount of time remaining.
            %   This implementation of progressbar is intended to be extremely simple to use
            % while providing a high quality user experience.
            %
            % Features:
            %   - Can add progressbar to existing m-files with a single line of code.
            %   - Supports multiple bars in one figure to show progress of nested loops.
            %   - Optional labels on bars.
            %   - Figure closes automatically when task is complete.
            %   - Only one figure can exist so old figures don't clutter the desktop.
            %   - Remaining time estimate is accurate even if the figure gets closed.
            %   - Minimal execution time. Won't slow down code.
            %   - Randomized color. When a programmer gets bored...
            %
            % Example Function Calls For Single Bar Usage:
            %   progressbar               % Initialize/reset
            %   progressbar(0)            % Initialize/reset
            %   progressbar('Label')      % Initialize/reset and label the bar
            %   progressbar(0.5)          % Update
            %   progressbar(1)            % Close
            %
            % Example Function Calls For Multi Bar Usage:
            %   progressbar(0, 0)         % Initialize/reset two bars
            %   progressbar('A', '')      % Initialize/reset two bars with one label
            %   progressbar('', 'B')      % Initialize/reset two bars with one label
            %   progressbar('A', 'B')     % Initialize/reset two bars with two labels
            %   progressbar(0.3)          % Update 1st bar
            %   progressbar(0.3, [])      % Update 1st bar
            %   progressbar([], 0.3)      % Update 2nd bar
            %   progressbar(0.7, 0.9)     % Update both bars
            %   progressbar(1)            % Close
            %   progressbar(1, [])        % Close
            %   progressbar(1, 0.4)       % Close
            %
            % Notes:
            %   For best results, call progressbar with all zero (or all string) inputs
            % before any processing. This sets the proper starting time reference to
            % calculate time remaining.
            %   Bar color is choosen randomly when the figure is created or reset. Clicking
            % the bar will cause a random color change.
            %
            % Demos:
            %     % Single bar
            %     m = 500;
            %     progressbar % Init single bar
            %     for i = 1:m
            %       pause(0.01) % Do something important
            %       progressbar(i/m) % Update progress bar
            %     end
            % 
            %     % Simple multi bar (update one bar at a time)
            %     m = 4;
            %     n = 3;
            %     p = 100;
            %     progressbar(0,0,0) % Init 3 bars
            %     for i = 1:m
            %         progressbar([],0) % Reset 2nd bar
            %         for j = 1:n
            %             progressbar([],[],0) % Reset 3rd bar
            %             for k = 1:p
            %                 pause(0.01) % Do something important
            %                 progressbar([],[],k/p) % Update 3rd bar
            %             end
            %             progressbar([],j/n) % Update 2nd bar
            %         end
            %         progressbar(i/m) % Update 1st bar
            %     end
            % 
            %     % Fancy multi bar (use labels and update all bars at once)
            %     m = 4;
            %     n = 3;
            %     p = 100;
            %     progressbar('Monte Carlo Trials','Simulation','Component') % Init 3 bars
            %     for i = 1:m
            %         for j = 1:n
            %             for k = 1:p
            %                 pause(0.01) % Do something important
            %                 % Update all bars
            %                 frac3 = k/p;
            %                 frac2 = ((j-1) + frac3) / n;
            %                 frac1 = ((i-1) + frac2) / m;
            %                 progressbar(frac1, frac2, frac3)
            %             end
            %         end
            %     end
            %
            % Author:
            %   Steve Hoelzer
            %
            % Revisions:
            % 2002-Feb-27   Created function
            % 2002-Mar-19   Updated title text order
            % 2002-Apr-11   Use floor instead of round for percentdone
            % 2002-Jun-06   Updated for speed using patch (Thanks to waitbar.m)
            % 2002-Jun-19   Choose random patch color when a new figure is created
            % 2002-Jun-24   Click on bar or axes to choose new random color
            % 2002-Jun-27   Calc time left, reset progress bar when fractiondone == 0
            % 2002-Jun-28   Remove extraText var, add position var
            % 2002-Jul-18   fractiondone input is optional
            % 2002-Jul-19   Allow position to specify screen coordinates
            % 2002-Jul-22   Clear vars used in color change callback routine
            % 2002-Jul-29   Position input is always specified in pixels
            % 2002-Sep-09   Change order of title bar text
            % 2003-Jun-13   Change 'min' to 'm' because of built in function 'min'
            % 2003-Sep-08   Use callback for changing color instead of string
            % 2003-Sep-10   Use persistent vars for speed, modify titlebarstr
            % 2003-Sep-25   Correct titlebarstr for 0% case
            % 2003-Nov-25   Clear all persistent vars when percentdone = 100
            % 2004-Jan-22   Cleaner reset process, don't create figure if percentdone = 100
            % 2004-Jan-27   Handle incorrect position input
            % 2004-Feb-16   Minimum time interval between updates
            % 2004-Apr-01   Cleaner process of enforcing minimum time interval
            % 2004-Oct-08   Seperate function for timeleftstr, expand to include days
            % 2004-Oct-20   Efficient if-else structure for sec2timestr
            % 2006-Sep-11   Width is a multiple of height (don't stretch on widescreens)
            % 2010-Sep-21   Major overhaul to support multiple bars and add labels
            %
            
            persistent progfig progdata lastupdate
        
            % Get inputs
            if nargin > 0
                input = varargin;
                ninput = nargin;
            else
                % If no inputs, init with a single bar
                input = {0};
                ninput = 1;
            end

            % If task completed, close figure and clear vars, then exit
            if input{1} == 1
                if ishandle(progfig)
                    delete(progfig) % Close progress bar
                end
                clear progfig progdata lastupdate % Clear persistent vars
                drawnow
                return
            end

            % Init reset flag 
            resetflag = false;

            % Set reset flag if first input is a string
            if ischar(input{1})                                
                resetflag = true;
            end

            % Set reset flag if all inputs are zero
            if input{1} == 0
                % If the quick check above passes, need to check all inputs
                if all([input{:}] == 0) && (length([input{:}]) == ninput)
                    resetflag = true;
                end
            end

            % Set reset flag if more inputs than bars
            if ninput > length(progdata)
                resetflag = true;
            end

            % If reset needed, close figure and forget old data
            if resetflag         
                if ishandle(progfig)                    
                    % KGD: I'd like to update the strings
                    if ischar(input{1}) && ishandle(progdata(1).proglabel)
                        set(progdata(1).proglabel, 'String', input{1})
                        return;
                    else                
                        delete(progfig) % Close progress bar
                    end
                end
                
                progfig = [];
                progdata = []; % Forget obsolete data
            end

            % Create new progress bar if needed
            if ishandle(progfig)
            else % This strange if-else works when progfig is empty (~ishandle() does not)

                % Define figure size and axes padding for the single bar case
                height = 0.03;
                width = height * 8;
                hpad = 0.02;
                vpad = 0.25;

                % Figure out how many bars to draw
                nbars = max(ninput, length(progdata));

                % Adjust figure size and axes padding for number of bars
                heightfactor = (1 - vpad) * nbars + vpad;
                height = height * heightfactor;
                vpad = vpad / heightfactor;

                % Initialize progress bar figure
                left = (1 - width) / 2;
                bottom = (1 - height) / 2;
                progfig = figure(...
                    'Units', 'normalized',...
                    'Position', [left bottom width height],...
                    'NumberTitle', 'off',...
                    'Resize', 'off',...
                    'MenuBar', 'none' );

                % Initialize axes, patch, and text for each bar
                left = hpad;
                width = 1 - 2*hpad;
                vpadtotal = vpad * (nbars + 1);
                height = (1 - vpadtotal) / nbars;
                for ndx = 1:nbars
                    % Create axes, patch, and text
                    bottom = vpad + (vpad + height) * (nbars - ndx);
                    progdata(ndx).progaxes = axes( ...
                        'Position', [left bottom width height], ...
                        'XLim', [0 1], ...
                        'YLim', [0 1], ...
                        'Box', 'on', ...
                        'ytick', [], ...
                        'xtick', [] );
                    progdata(ndx).progpatch = patch( ...
                        'XData', [0 0 0 0], ...
                        'YData', [0 0 1 1] );
                    progdata(ndx).progtext = text(0.99, 0.5, '', ...
                        'HorizontalAlignment', 'Right', ...
                        'FontUnits', 'Normalized', ...
                        'FontSize', 0.7 );
                    progdata(ndx).proglabel = text(0.01, 0.5, '', ...
                        'HorizontalAlignment', 'Left', ...
                        'FontUnits', 'Normalized', ...
                        'FontSize', 0.85);
                    if ischar(input{ndx})
                        set(progdata(ndx).proglabel, 'String', input{ndx})
                        input{ndx} = 0;
                    end

                    % Set callbacks to change color on mouse click
                    %set(progdata(ndx).progaxes, 'ButtonDownFcn', {@changecolor, progdata(ndx).progpatch})
                    %set(progdata(ndx).progpatch, 'ButtonDownFcn', {@changecolor, progdata(ndx).progpatch})
                    %set(progdata(ndx).progtext, 'ButtonDownFcn', {@changecolor, progdata(ndx).progpatch})
                    %set(progdata(ndx).proglabel, 'ButtonDownFcn', {@changecolor, progdata(ndx).progpatch})

                    % Pick a random color for this patch
                    changecolor([], [], progdata(ndx).progpatch)

                    % Set starting time reference
                    if ~isfield(progdata(ndx), 'starttime') || isempty(progdata(ndx).starttime)
                        progdata(ndx).starttime = clock;
                    end
                end

                % Set time of last update to ensure a redraw
                lastupdate = clock - 1;

            end

            % Process inputs and update state of progdata
            for ndx = 1:ninput
                if ~isempty(input{ndx})
                    progdata(ndx).fractiondone = input{ndx};
                    progdata(ndx).clock = clock;
                end
            end

            % Enforce a minimum time interval between graphics updates
            myclock = clock;
            if abs(myclock(6) - lastupdate(6)) < 0.01 % Could use etime() but this is faster
                return
            end

            % Update progress patch
            for ndx = 1:length(progdata)
                set(progdata(ndx).progpatch, 'XData', ...
                    [0, progdata(ndx).fractiondone, progdata(ndx).fractiondone, 0])
            end

            % Update progress text if there is more than one bar
            if length(progdata) > 1
                for ndx = 1:length(progdata)
                    set(progdata(ndx).progtext, 'String', ...
                        sprintf('%1d%%', floor(100*progdata(ndx).fractiondone)))
                end
            end

            % Update progress figure title bar
            if progdata(1).fractiondone > 0
                runtime = etime(progdata(1).clock, progdata(1).starttime);
                timeleft = runtime / progdata(1).fractiondone - runtime;
                timeleftstr = sec2timestr(timeleft);
                titlebarstr = sprintf('%2d%%    %s remaining', ...
                    floor(100*progdata(1).fractiondone), timeleftstr);
            else
                titlebarstr = ' 0%';
            end
            set(progfig, 'Name', titlebarstr)

            % Force redraw to show changes
            drawnow

            % Record time of this update
            lastupdate = clock;
            
            % ----------------------------------------------------------------
            function changecolor(h, e, progpatch) %#ok<INUSL>
                % Change the color of the progress bar patch

                % Prevent color from being too dark or too light
                colormin = 1.5;
                colormax = 2.8;

                thiscolor = rand(1, 3);
                while (sum(thiscolor) < colormin) || (sum(thiscolor) > colormax)
                    thiscolor = rand(1, 3);
                end

                set(progpatch, 'FaceColor', thiscolor)
            end
            
            % ------------------------------------------------------------------------------
            function timestr = sec2timestr(sec)
                % Convert a time measurement from seconds into a human readable string.

                % Convert seconds to other units
                w = floor(sec/604800); % Weeks
                sec = sec - w*604800;
                d = floor(sec/86400); % Days
                sec = sec - d*86400;
                h = floor(sec/3600); % Hours
                sec = sec - h*3600;
                m = floor(sec/60); % Minutes
                sec = sec - m*60;
                s = floor(sec); % Seconds

                % Create time string
                if w > 0
                    if w > 9
                        timestr = sprintf('%d week', w);
                    else
                        timestr = sprintf('%d week, %d day', w, d);
                    end
                elseif d > 0
                    if d > 9
                        timestr = sprintf('%d day', d);
                    else
                        timestr = sprintf('%d day, %d hr', d, h);
                    end
                elseif h > 0
                    if h > 9
                        timestr = sprintf('%d hr', h);
                    else
                        timestr = sprintf('%d hr, %d min', h, m);
                    end
                elseif m > 0
                    if m > 9
                        timestr = sprintf('%d min', m);
                    else
                        timestr = sprintf('%d min, %d sec', m, s);
                    end
                else
                    timestr = sprintf('%d sec', s);
                end
                
            end


            
        end % ``progressbar``
        
        function b = regress_func(Y, x, has_missing)
            % Regress vector ``x`` onto the columns in matrix ``Y`` one at a time.
            % Return the vector of regression coefficients, one for each column in ``Y``.
            % There may be missing data in ``Y``, but never in ``x``.  
            % The ``x`` vector *must* be a column vector.

            % Supply the true/false indicator, ``has_missing`` to avoid slower calculations
            % for missing data.


            [Ny, K] = size(Y);
            Nx = numel(x);

            if Ny == Nx                  % Case A: b' = (x'Y)/(x'x): (1xN)(NxK) = (1xK)
                if not(has_missing)
                    b = (Y'*x)/(x'*x);
                    return
                end
                b = zeros(K, 1);
                for k = 1:K
                    keep = ~isnan(Y(:,k));
                    b(k) = sum(x(keep, 1) .* Y(keep, k));
                    denom = norm(x(keep))^2;
                    if abs(denom) > eps
                        b(k) = b(k) / denom;
                    end
                end
            elseif K == Nx
                % Case B:  b = (Yx)/(x'x): (NxK)(Kx1) = (Nx1)
                % Regressing x onto rows in Y, storing results in column vector "b"
                if not(has_missing)
                    b = (Y*x)/(x'*x);
                    return
                end

                b = zeros(Ny, 1);    
                for n = 1:Ny
                    keep = ~isnan(Y(n, :));
                    b(n) = sum(x(keep)' .* Y(n, keep)); %    sum(x(:,0) * np.nan_to_num(Y(n,:)));
                    % TODO(KGD): check: this denom is usually(always?) equal to 1.0
                    denom = norm(x(keep))^2;
                    if abs(denom) > eps
                        b(n) = b(n) / denom;
                    end

                end
            end
        end  % ``regress_func``
        
        function out = ssq(X, axis)

            % A function than calculates the sum of squares of a matrix (not array!),
            % skipping over any NaN (missing) data.
            %
            % If ``axis`` is not specified, it will sum over the entire array and
            % return a scalar value.  If ``axis`` is specified, then the output is
            % usually a vector, with the sum of squares taken along that axis.
            %
            % If a complete dimension has missing values, then ssq will return 0.0 for
            % that sum of squares.
            %
            % Relies on nansum.m
            
            function y = nansum(x, dim)
                % FORMAT: Y = NANSUM(X,DIM)
                % 
                %    Sum of values ignoring NaNs
                %
                %    This function enhances the functionality of NANSUM as distributed in
                %    the MATLAB Statistics Toolbox and is meant as a replacement (hence the
                %    identical name).  
                %
                %    NANSUM(X,DIM) calculates the mean along any dimension of the N-D array
                %    X ignoring NaNs.  If DIM is omitted NANSUM averages along the first
                %    non-singleton dimension of X.
                %
                %    Similar replacements exist for NANMEAN, NANSTD, NANMEDIAN, NANMIN, and
                %    NANMAX which are all part of the NaN-suite.
                %
                %    See also SUM

                % -------------------------------------------------------------------------
                %    author:      Jan Gläscher
                %    affiliation: Neuroimage Nord, University of Hamburg, Germany
                %    email:       glaescher@uke.uni-hamburg.de
                %    
                %    $Revision: 1.2 $ $Date: 2005/06/13 12:14:38 $
                if isempty(x)
                    y = [];
                    return
                end

                if nargin < 2
                    dim = min(find(size(x)~=1));
                    if isempty(dim)
                        dim = 1;
                    end
                end

                % Replace NaNs with zeros.
                nans = isnan(x);
                x(isnan(x)) = 0; 

                % Protect against all NaNs in one dimension
                count = size(x,dim) - sum(nans,dim);
                i = find(count==0);

                y = sum(x,dim);
                y(i) = NaN;
            end

            if nargin == 1
                out = nansum(X(:).*X(:));
            else
                out = nansum(X.*X, axis);
            end
            out(isnan(out)) = 0.0;
            
        end % ``ssq``
        
        function out = zeroexp(shape, varargin)

            % Will return an array of zeros of the given ``shape``.  Or if given an
            % existing array ``prev`` will expand it to the new ``shape``, maintaining
            % the existing elements, but adding zeros to the new elements.  Only
            % one of the elements in ``shape`` may be different from size(prev)

            out = zeros(shape);
            if nargin == 3
                % If "true" then expand it with NaN's rather than zeros
                if varargin{2}
                    out = out .* NaN;
                end
            end
            if nargin == 2 && not(isempty(varargin{1}))
                prev = varargin{1};
                prev_shape = size(prev);
            else
                return;
            end


            % Which dimension is to be expanded?
            which_dim = find(shape-prev_shape);
            if numel(which_dim) > 1
                error('The new shape must have all dimensions, except one, as the same size.');
            end
            % If the dimensions are the same as required, just pass the input as
            % the output
            if isempty(which_dim)
                out = prev;
            else
                idx = cell(1,numel(shape));
                for k = 1:numel(shape)
                    idx{k} = ':';
                end
                idx{which_dim} = 1:prev_shape(which_dim);        
                subsarr = struct('type', '()', 'subs', {idx});
                subsasgn(out, subsarr, prev);

            end   
        end % ``zerosexp``
        
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
            
            if idx == 0
                spe_data = [];
                return
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
            
            if idx == 0
                return
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
                if not(isempty(labels))
                    for n = 1:numel(breaks)
                         text(breaks(n), top, strtrim(labels{n}), 'Rotation', 0,...
                             'FontSize', 10, 'HorizontalAlignment', 'center');
                    end
                end
                %set(ax, 'TickLength', [0 0], 'XLim', [breaks(1)-0.5 breaks(end)+0.5])


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
