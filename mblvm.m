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
        function self = mblvm(varargin)
            
            % Process model options, if provided
            % -----------------------
            self.opt = lvm_opt('default');

            % Build the model if ``A``, is specified as a
            % numeric value.  If it is a structure of 
            % options, then override any default settings with those.            

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
        end % ``disp``
        
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
                
            preprocess_blocks(self);       % superclass method
            calc_model(self, requested_A); % method must be subclassed
        end % ``build``
        
        function state = apply(self, new)
            % Apply the multiblock latent variable model to new data.
            % * preprocess the data if it has not been already
            
            newb = cell(1, numel(new)/2);
            Nnew = 0;
            for b = 1:numel(new)
                if mod(b, 2) ~= 0
                    block_name = new{b};
                    if not(strcmpi(block_name, self.blocks{b}.name))
                        error('lvm:lvm', 'Block names on the new data do not agree with training data.') 
                    end                    
                else 
                    newb{b/2} = block(new{b});                    
                    Nnew = max(Nnew, newb{b/2}.N);
                end
            end        
            
            for b = 1:self.B
                newb{b} = self.blocks{b}.preprocess(newb{b}, self.PP{b});
            end
            
            % Initialize the states (this could go in another function later)
            state = struct;
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
            state.stats.R2 = cell(1, self.B);
            state.stats.initial_ssq = cell(1, self.B);
            state.stats.super.T2 = ones(Nnew, 1) .* NaN;
            state.stats.super.SPE = ones(Nnew, 1) .* NaN;
            state.stats.super.R2 = ones(Nnew, 1) .* NaN;
            state.stats.initial_ssq_total = ones(Nnew, 1) .* NaN;
            for b = 1:self.B
                state.stats.R2{b} = ones(Nnew, 1) .* NaN;
            end            
            state = apply_model(self, newb, state); % method must be subclassed
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
            %self.S = cell(1,nb);

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
            % Superblock's loadings (used in PCA only)
            % B x A
            self.super.P = [];    
            % Superblocks's weights (used in PLS only)
            % B x A
            self.super.W = [];   
            % R2 for each component for the overall model
            self.super.stats.R2 = [];
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
        end
        
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
                self.P{b} = zeroexp([dblock.K, A], self.P{b});  % loadings; .C for PLS
                self.T{b} = zeroexp([dblock.N, A], self.T{b});  % block scores, .U for PLS                
                self.W{b} = zeroexp([dblock.K, A], self.W{b});  % PLS weights
                self.R{b} = zeroexp([dblock.K, A], self.R{b});  % PLS weights
                self.C{b} = zeroexp([dblock.K, A], self.C{b});  % PLS Y-space loadings
                self.U{b} = zeroexp([dblock.N, A], self.U{b});  % PLS Y-space scores
               %self.S{b} = zeroexp([1, A], self.S{b});         % score scaling factors
                
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
                    self.stats{b}.R2k_a = [];  % not cumulative !
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
                self.stats{b}.R2k_a = zeroexp([dblock.K, A], self.stats{b}.R2k_a);
                
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
            
            % T2, using all components 1:A, in the superblock
            self.super.T2 = zeroexp([self.N, A], self.super.T2);
            
            % Limits for superscore entities
            self.super.lim.t = zeroexp([1, A], self.super.lim.t);
            self.super.lim.T2 = zeroexp([1, A], self.super.lim.T2);            
            self.super.lim.SPE = zeroexp([1, A], self.super.lim.SPE);
            
            % Statistics in the superblock
            self.super.stats.R2 = zeroexp([1, A], self.super.stats.R2);
            self.super.stats.SSQ_exp = zeroexp([1, A], self.super.stats.R2);
            % The VIP's for each block, and the VIP calculation factor
            self.super.stats.VIP = zeroexp([self.B, 1], self.super.stats.VIP);
            %self.super.stats.VIP_f = cell({});
            
            % Give the subclass the chance to expand storage, if required
            self.expand_storage(A);
        end % ``initialize_storage``
        
        % TODO(KGD): preprocessing should be moved to its own class
        %            where it can handle "self" preprocessing and on 
        %            "other" blocks; and more general preprocessing options.
        function self = preprocess_blocks(self)
            % Preprocesses each block.            
            for b = 1:self.B
                if ~self.PP{b}.is_preprocessed
                    [self.blocks{b}, PP_block] = self.blocks{b}.preprocess();
                    self.PP{1}.is_preprocessed = true;
                    self.PP{1}.mean_center = PP_block.mean_center;
                    self.PP{1}.scaling = PP_block.scaling;                    
                end                
            end
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

    end % end: methods (sealed)
    
    % Subclasses must redefine these methods
    methods (Abstract=true)
        calc_model(self, A)      % fit the model to data in ``self``
        apply_model(self, other) % applies the model to ``new`` data
        expand_storage(self, A)
    end % end: methods (abstract)
    
    methods (Sealed=true, Static=true)
        function out = mahalanobis_distance(T)
            % TODO(KGD): calculate this in a smarter way. Can create unnecessarily
            % large matrices
            n = size(T, 1);
            out = diag(T * inv((T'*T)/(n-1)) * T'); %#ok<MINV>
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
            
        function y = chi2inv(p, nu)
            % http://www.atmos.washington.edu/~wmtsa/
            %
            % Copyright (c) 2003,2004,2005, Charles R. Cornish
            % All rights reserved.
            % 
            % Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
            % 
            %     * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
            %     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
            %     * Neither the name of the owner nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
            % 
            % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
         
            p = 1 - p;
            
            % computes the  abscissae y for given probability p (upper tail)
            % for a chi square distribution with nu >=2 degrees
            % of freedom, NOT NECESSARILY AN INTEGER. Based on
            % Goldstein, R.B., Collected Algorithms for Computer Machinery,
            % 16, 483-485.
            %
            %  p: real col vector of right-tail probs
            %  nu: possibly non-integer deg of freedom
            %  y: real column vector of abscissae
            %
            % Author: Andrew Walden


            c=[1.565326e-3,1.060438e-3,-6.950356e-3,-1.323293e-2,2.277679e-2,...
            -8.986007e-3,-1.513904e-2,2.530010e-3,-1.450117e-3,5.169654e-3,...
            -1.153761e-2,1.128186e-2,2.607083e-2,-0.2237368,9.780499e-5,-8.426812e-4,...
            3.125580e-3,-8.553069e-3,1.348028e-4,0.4713941,1.0000886];

            a=[1.264616e-2,-1.425296e-2,1.400483e-2,-5.886090e-3,...
            -1.091214e-2,-2.304527e-2,3.135411e-3,-2.728484e-4,...
            -9.699681e-3,1.316872e-2,2.618914e-2,-0.2222222,...
            5.406674e-5,3.483789e-5,-7.274761e-4,3.292181e-3,...
            -8.729713e-3,0.4714045,1.0];

            %  I HAVE RESTRICTED TO Nu>=2 SO THAT CAN MAKE Nu REAL.
            if (nu < 2) 
              error('QChisq ERROR: degrees of freedom < 2') 
            end 
            if nu==2
                y = -2.*log(p);
            else % nu >2
                f=nu;
                f1=1./f;
                t = sqrt(2).*erfinv(1-2.*p); % upper tail area p
                f2=sqrt(f1).*t;
                if nu < 2.+fix(4.*abs(t))
                    y=(((((((c(1).*f2+c(2)).*f2+c(3)).*f2+c(4)).*f2...
                    +c(5)).*f2+c(6)).*f2+c(7)).*f1+((((((c(8)+c(9).*f2).*f2...
                    +c(10)).*f2+c(11)).*f2+c(12)).*f2+c(13)).*f2+c(14))).*f1...
                    +(((((c(15).*f2+c(16)).*f2+c(17)).*f2+c(18)).*f2...
                    +c(19)).*f2+c(20)).*f2+c(21);
                else
                    y=(((a(1)+a(2).*f2).*f1+(((a(3)+a(4).*f2).*f2...
                   +a(5)).*f2+a(6))).*f1+(((((a(7)+a(8).*f2).*f2+a(9)).*f2...
                   +a(10)).*f2+a(11)).*f2+a(12))).*f1+(((((a(13).*f2...
                   +a(14)).*f2+a(15)).*f2+a(16)).*f2+a(17)).*f2.*f2...
                   +a(18)).*f2+a(19);
                end
                y = (y.^3).*f;
            end
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
               xn = my_norminv(p(k));
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
            n = numel(score_column);
            alpha = (1-levels) ./ 2.0;
            n_ppf = mblvm.tinv(1-alpha, n-1);            

            % NORMAL DISTRIBUTION ASSUMPTION
            % 
            % n_ppf = mblvm.norminv(1-alpha);
            
            limits = n_ppf * std(score_column);
        end

    end % end: methods (sealed and static)
    
end % end classdef

%-------- Helper functions (usually used in 2 or more places). May NOT use ``self``

% Calculates the leverage 

