% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for defining, building and manipulating latent variable models.


% Create a handle class
% This is the highest level latent variable model class
classdef mblvm < handle
    properties
        model_type = ''; % for convenience, shouldn't be used in code; displays model type
        blocks = {};     % a cell array of data blocks; last block is always "Y" (can be empty)
        model = struct();% model structure        
        A = 0;           % number of latent variables
        B = 0;           % number of blocks
        opt = struct();  % model options
        stats = struct();% Model statistics (time, iterations)
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
            
            % Initialize some structures
            self.stats.timing = [];
            self.stats.itern = [];          

            
        end % ``lvm``
        
        function out = get.B(self)
            % Purely a convenience function to get ".B"
            out = numel(self.blocks);
        end
            
        function out = iter_terminate(lv_prev, lv_current, itern, tolerance, model, conditions)
            % The latent variable algorithm is terminated when any one of these
            % conditions is True
            %  #. scores converge: the norm between two successive iterations
            %  #. a max number of iterations is reached
            score_tol = norm(lv_prev - lv_current);                       
            conditions.converged = score_tol < tolerance;
            conditions.max_iter = itern > model.opt.max_iter;            
            
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
        function self = build(self, varargin)
            % Build the multiblock latent variable model.
            % * preprocess the data if it has not been already
          
            if nargin==2 && isa(varargin{1}, 'numeric')
                given_A = varargin{1};
            else
                given_A = 0;
            end

            requested_A = max([self.opt.min_lv, 0, given_A]);            
            self.stats.timing = zeroexp([requested_A, 1], self.stats.timing);
            self.stats.itern = zeroexp([requested_A, 1], self.stats.itern);
            
            self = preprocess_blocks(self, requested_A);  % superclass method
            self = calc_model(self, requested_A);         % must be subclassed
            self = calc_statistics(self); 
            self = calc_limits(self);    
            
        end % ``build``
        
        function self = preprocess_blocks(self, A)
            % Preprocesses each block. Resizes storage for A components.
            
            
            % TODO: handle the case where the model is shrunk or grown to 
            %       a different A value.
            
            for b = 1:self.B
                if ~self.blocks{b}.is_preprocessed
                    self.blocks{b} = self.blocks{b}.preprocess();
                end
                % Resize the storage for ``A`` components
                self.blocks{b} = self.blocks{b}.initialize_storage(A);
            end
        end % ``preprocess_blocks``
    end % methods (sealed)
    
    % Subclasses must redefine these methods
    methods (Abstract=true)
        self = calc_model(self, A)
    end % methods (abstract)
    
end % end classdef

%-------- Helper functions (usually used in 2 or more places). May NOT use ``self``


