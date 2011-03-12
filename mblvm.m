% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for defining, building and manipulating latent variable models.


% Create a handle class
% This is the highest level latent variable model class
classdef mblvm < handle
    properties
        model_type = '';    % for convenience, shouldn't be used in code; displays model type
        blocks = {};        % a cell array of data blocks; last block is always "Y" (can be empty)
        A = 0;              % number of latent variables
        B = 0;              % number of blocks
        
        
        opt = struct();     % model options
        stats = cell({});   % Model statistics for each block
        model = cell({});   % Model-related statistics (timing, iterations, risk)
        lim = cell({});     % Model limits
        
        % Model parameters for each block (each cell entry is a block)
        P = cell({});       % Block loadings, P
        T = cell({});       % Block scores, T        
        S = cell({});       % Std deviation of the scores
        W = cell({});       % Block weights (for PLS only)
        R = cell({});       % Block weights (W-star matrix, for PLS only) 
        C = cell({});       % Block loadings (for PLS only)
        U = cell({});       % Block scores (for PLS only) 
        beta = cell({});    % Beta-regression coeffients 
        PP = cell({});      % Preprocess model parameters 
        
        % Specific to batch models
        T_j = [];    % Instantaneous scores
        error_j = [];% Instantaneous errors
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
            self = self.create_storage();
             
        end % ``lvm``
        
        function out = get.B(self)
            % Purely a convenience function to get ".B"
            out = numel(self.blocks);
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
    end % methods (ordinary)
    
    % Subclasses may not redefine these methods
    methods (Sealed=true)
        function self = create_storage(self)
            
            % General model statistics
            self.model.stats.timing = [];
            self.model.stats.itern = [];     
            
            nb = self.B;
            
            % Latent variable parameters
            self.P = cell(1,nb);
            self.T = cell(1,nb);
            self.S = cell(1,nb);
            self.W = cell(1,nb);
            self.R = cell(1,nb);
            self.C = cell(1,nb);
            self.U = cell(1,nb);

            % Block preprocessing 
            self.PP = cell(1,nb);

            self.stats = cell(1,nb);
            self.lim = cell(1,nb);
            
        end
        
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
            self = self.initialize_storage(requested_A);                
                
            self = preprocess_blocks(self);               % superclass method
            self = calc_model(self, requested_A);         % must be subclassed
            self = calc_statistics_and_limits(self);      % must be subclassed           
        end % ``build``
        
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
                self = create_storage(self);
            end
            
            self.model.stats.timing = zeroexp([A, 1], self.model.stats.timing);
            self.model.stats.itern = zeroexp([A, 1], self.model.stats.itern);
            
            for b = 1:self.B
                dblock = self.blocks{b};
                self.P{b} = zeroexp([dblock.K, A], self.P{b});  % loadings; .C for PLS
                self.T{b} = zeroexp([dblock.N, A], self.T{b});  % block scores, .U for PLS
                self.S{b} = zeroexp([1, A], self.S{b});         % score scaling factors
                self.W{b} = zeroexp([dblock.K, A], self.W{b});  % PLS weights
                self.R{b} = zeroexp([dblock.K, A], self.R{b});  % PLS weights
                self.C{b} = zeroexp([dblock.K, A], self.C{b});  % PLS Y-space loadings
                self.U{b} = zeroexp([dblock.N, A], self.U{b});  % PLS Y-space scores
                
                % Block preprocessing options: resets them
                if numel(self.PP{b}) == 0
                    self.PP{b} = struct('mean_center', [], 'scaling', []);                    
                end

                % Calculated statistics for this block
                if numel(self.stats{b}) == 0
                    self.stats{b}.SPE = [];
                    self.stats{b}.SPE_j = [];

                    self.stats{b}.start_SS_col = [];
                    self.stats{b}.deflated_SS_col = [];                
                    self.stats{b}.R2k_a = [];
                    self.stats{b}.R2k_cum = [];
                    self.stats{b}.R2_a = [];
                    self.stats{b}.R2 = [];
                    self.stats{b}.VIP_a = [];
                    self.stats{b}.VIP = [];

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

                self.stats{b}.SPE = zeroexp([dblock.N, A], self.stats{b}.SPE);
                % Instantaneous SPE limit using all A components (batch models)
                self.stats{b}.SPE_j = zeroexp([dblock.N, dblock.J], self.stats{b}.SPE_j, true);


                % R^2 per variable, per component; cumulative R2 per variable
                % Baseline value for all R2 calculations: before any components are
                % extracted, but after the data have been preprocessed.
                self.stats{b}.start_SS_col = zeroexp([1, dblock.K], self.stats{b}.start_SS_col);
                % Used in cross-validation calculations: ssq of each column,
                % per component, after deflation with the a-th component.
                self.stats{b}.deflated_SS_col = zeroexp([dblock.K, A], self.stats{b}.deflated_SS_col);
                self.stats{b}.R2k_a = zeroexp([dblock.K, A], self.stats{b}.R2k_a);
                self.stats{b}.R2k_cum = zeroexp([dblock.K, A], self.stats{b}.R2k_cum);
                % R^2 per block, per component; cumulate R2 for the block
                self.stats{b}.R2_a = zeroexp([A, 1], self.stats{b}.R2_a);
                self.stats{b}.R2 = zeroexp([A, 1], self.stats{b}.R2);

                % VIP value (only calculated for X-blocks); only last column is useful
                self.stats{b}.VIP_a = zeroexp([dblock.K, A], self.stats{b}.VIP_a);
                self.stats{b}.VIP = zeroexp([dblock.K, 1], self.stats{b}.VIP);

                % Overall T2 value for each observation
                self.stats{b}.T2 = zeroexp([dblock.N, 1], self.stats{b}.T2);
                % Instantaneous T2 limit using all A components (batch models)
                self.stats{b}.T2_j = zeroexp([dblock.N, dblock.J], self.stats{b}.T2_j);

                % Modelling power = 1 - (RSD_k)/(RSD_0k)
                % RSD_k = residual standard deviation of variable k after A PC's
                % RSD_0k = same, but before any latent variables are extracted
                % RSD_0k = 1.0 if the data have been autoscaled.
                self.stats{b}.model_power = zeroexp([1, dblock.K], self.stats{b}.model_power);

                % Actual limits for the block: to be calculated later on
                % ---------------------------
                % Limits for the (possibly time-varying) scores
                %siglevels = {'95.0', '99.0'};
                self.lim{b}.t = zeroexp([1, A], self.lim{b}.t);
                self.lim{b}.t_j = zeroexp([dblock.J, A], self.lim{b}.t_j, true); 

                % Hotelling's T2 limits using A components (column)
                % (this is actually the instantaneous T2 limit,
                % but we don't call it that, because at time=J the T2 limit is the
                % same as the overall T2 limit - not so for SPE!).
                self.lim{b}.T2 = zeroexp([1, A], self.lim{b}.T2);            

                % SPE limits for the block and instaneous (i.e. time-varying) limits
                % Overall SPE limit using for ``a`` components (column)
                self.lim{b}.SPE = zeroexp([1, A], self.lim{b}.SPE);

                % SPE instantaneous limits using all A components
                self.lim{b}.SPE_j = zeroexp([dblock.J, 1], self.lim{b}.SPE_j);

            end
  
            % Give the subclass the chance to expand storage, if required
            self.expand_storage(A)
        end % ``initialize_storage``
        
        function self = preprocess_blocks(self)
            % Preprocesses each block.            
            for b = 1:self.B
                if ~self.blocks{b}.is_preprocessed
                    self.blocks{b} = self.blocks{b}.preprocess();
                end                
            end
        end % ``preprocess_blocks``
        
    end % methods (sealed)
    
    % Subclasses must redefine these methods
    methods (Abstract=true)
        self = calc_model(self, A)
        self = calc_statistics_and_limits(self);
        self = expand_storage(self, A)
    end % methods (abstract)
    
end % end classdef

%-------- Helper functions (usually used in 2 or more places). May NOT use ``self``


