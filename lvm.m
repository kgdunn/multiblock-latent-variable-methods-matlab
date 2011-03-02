% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for defining, building and manipulating latent variable models.

function self = lvm(blocks, varargin)
% Creates a latent variable model (LVM) object.
%
% SYNTAX: the only requirements are
%    * that 'Y' be used for PLS models
%    * that a block should be pre-made into a batch block for batch models
%
% PCA model: lvm({'mydata', x_block}) 
% PLS model: lvm({'predictors', x_block, 'Y', y_block}) 
% Multiblock PCA: lvm({'X1', X1, 'X2', X2, ....})
% Multiblock PLS: lvm({'X1', X1, 'X2', X2, ...., 'Y', Y})
%
% Batch PCA: batch_X = block(X, 'batch', ....)
%            lvm({'my batch data', batch_X})
%
% Batch PCA with a Z matrix:
%            batch_X = block(X, 'batch', ....)
%            lvm({'initial', Z, 'my batch data', batch_X})
%        or  lvm({'my batch data', batch_X, 'initial', Z})  % in any order
%
% Batch PLS:
%            batch_X = block(X, 'batch', ....)
%            lvm({'my batch data', batch_X, 'Y', y_block})
%
% Multiblock, batch PLS model: 
%            lvm({'Z1', Z1, 'Z2', Z2, 'X', batch_X, 'Y', y_block}) 
  
    if mod(numel(blocks), 2) ~= 0
        error('lvm:lvm', 'First input must be provided as a cell array of pairs, e.g. {''X'', x_data, ''Y'', y_data}')    
    end
    if iscell(blocks)
        % FUTURE(KGD): use a dictionary, or MATLAB/Java hashtable for the blocks
        out = cell(1, numel(blocks)/2);
        model_type = 'PCA';
        for b = 1:numel(blocks)
            if mod(b, 2) ~= 0
                block_name = blocks{b};
                if not(ischar(block_name))
                    error('lvm:lvm', 'Block name must be a character string.') 
                end
                if strcmpi(block_name, 'y')
                    model_type = 'PLS';
                end
            else 
                out{b/2} = block(blocks{b});
                if strcmp(out{b/2}.name_type, 'default')
                    out{b/2}.name = block_name;
                    out{b/2}.name_type = 'given';
                end
            end
        end
    else
        error('lvm:lvm', 'Please provide data in a cell arrays of pairs.')    
    end

    % Just create an empty (shell) class instance
    if strcmpi(model_type, 'pca')
        self = mbpca(varargin{:});
    elseif strcmpi(model_type, 'pls')
        self = mbpls(varargin);
    end
 
    % Now set the data blocks:
    self.blocks = out;
    
    if numel(out) > 2
        self.model_type = ['MB-', model_type];
    else
        self.model_type = model_type;
    end    

    if self.opt.build_now
       self = build(self);
    end            

        