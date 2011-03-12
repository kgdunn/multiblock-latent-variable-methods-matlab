% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for creating data blocks. A block is the smallest subset of data in a
% model.

function out = block(varargin)
% SYNTAX
% ======
% block(data)
% block(data, block_name)
% block(data, {'row_labels', row_names})
% block(data, {'col_labels', tag_names})
% block(data, {'row_labels', row_names}, {'col_labels', tag_names})
% block(data, block_name, {'row_labels', row_names}, {'col_labels', tag_names})
% 
% Examples
% --------
%
% block(randn(10,3))
% block(randn(10,3), 'Random data')
% row_names = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'Ten'}
% tag_names = {'First', 'Second', 'Third'}
% block(randn(10,3), 'Random data', {'row_labels', row_names}, ...
%                                   {'col_labels', tag_names})
%
% Batch data
% -----------
% block(data, block name, {'nBatches', num batches})
% block(data, block name, {'batch_tag_names', tag_names}, ... <-- name of each tag in batch  
%                         {'time_names', time_names}, ...     <-- cell array, or vector of integers
%                         {'batch_names', batch_names})       <-- name of each batch

% First argument: the data
rest = {};
if nargin == 0
    given_data = [];
else
    given_data = varargin{1};
end
if isa(given_data, 'block_base')
    % TODO(KGD): later on, continue processing this block; the user might want
    %            to add features, such as labels, etc to the existing block
    out = given_data;
    return    
end            

% Optional second argument: block name
given_name = [];
if nargin > 1 && ischar(varargin{2})
    given_name = varargin{2};
    rest = varargin(3:end);
elseif nargin > 1 && ~ischar(varargin{2})
    rest = varargin(2:end);
end

% Is it a batch block?
is_batch = false;
for i = 1:numel(rest)
    key = rest{i}{1};
    if strcmpi(key, 'nbatches')
        is_batch = true;
    elseif strcmpi(key, 'batch_tag_names')
        is_batch = true;
    elseif strcmpi(key, 'batch_names')
        is_batch = true;
    elseif strcmpi(key, 'time_names')
        is_batch = true;
    end 
end

% Default source
stack = dbstack(1);
if numel(stack) == 0
    source = '<MATLAB command window>';
else
    source = [stack(1).file, ', line ', num2str(stack(1).line)];
end
if is_batch
    out = block_batch(given_data, given_name, source, rest{:});
else
    out = block_base(given_data, given_name, source, rest{:});
end
