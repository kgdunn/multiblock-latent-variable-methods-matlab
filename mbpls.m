% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Multi-block PLS models

% Derived from the general class of "multiblock latent variable models", mblvm
classdef mbpls < mblvm
    methods 
        function self = mbpls(varargin)
            self = self@mblvm(varargin{:});            
        end % ``mbpls``        
        
        % Superclass abstract method implementation
        function self = expand_storage(self, varargin)
            % Do nothing: super-class methods are good enough
        end % ``expand_storage``
        
        % Superclass abstract method implementation
        function self = calc_model(self, A)
            
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
        function [t_a, p_a, itern] = single_block_PCA(dblock, self, a, has_missing)
            
        end
    end % end methods (static)
    
end % end classdef