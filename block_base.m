% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for creating data blocks. A block is the smallest subset of data in a
% model.

classdef block_base < handle
    
    % Class attributes
    properties
        data = false;           % Block's data
        has_missing = false;    % {true, false}
        mmap = false;           % Missing data map
        is_preprocessed = false;% {true, false}
        name = '';
        name_type = 'auto';     % {'auto', 'given'}
        labels = {};            % Cell array: rows are the modes; columns are sets of labels
                                % 2 x 3: row labels and columns labels, with
                                % up to 3 sets of labels for each dimension
        N = 0;                  % Number of observations
        K = 0;                  % Number of variables (columns)
    end
    
    methods
        function self = block_base(given_data, block_name, varargin)
            % SYNTAX:
            % block_base(data, block_name, variable arguments in cell arrays)
            
            % Store whatever data was given us
            self.data = given_data;
            
            [self.N, self.K] = size(given_data);
            self.labels = cell(ndims(given_data), 0);    % 0-columns of labels
            
            % Missing data handling
            self.mmap = false;            
            missing_map = ~isnan(given_data);   % 0=missing, 1=present
            if all(missing_map)
                self.has_missing = false;
            else
                self.has_missing = true;
                self.mmap = missing_map;
            end
            
            % Second argument: block name
            if isempty(block_name)
                self.name = ['block-', num2str(self.N), '-', num2str(self.K)];
                self.name_type = 'auto';
            else
                self.name = block_name;
                self.name_type = 'given';
            end
            
            % Third and subsequent arguments: 
            if nargin > 2
                for i = 1:numel(varargin)
                    key = varargin{i}{1};
                    value = varargin{i}{2};
                        
                    % ``row_labels``
                    if strcmpi(key, 'row_labels')
                        self.add_labels(1, value);
                        
                    % ``col_labels``
                    elseif strcmpi(key, 'col_labels')
                        self.add_labels(2, value);
                        
                    end
                end
            end
            
            
        end
  
        function add_labels(self, dim, to_add)           
            added = false;
            for k = 1:numel(self.labels(dim, :))
                if isempty(self.labels{dim, k})
                    self.labels{dim, k} = to_add;
                    return
                end
            end
            
            if ~added
                self.labels{dim, end+1} = to_add;
            end
            
        end
        
        function disp(self)
            % Displays a text summary of the block
            fprintf('%s: %d observations and %d variables\n', self.name, self.N, self.K)
            
            if self.has_missing
                fprintf('* Has missing data\n')
            else
                fprintf('* Has _no_ missing data\n')
            end
            if self.is_preprocessed
                fprintf('* Has been preprocessed\n')
            else
                fprintf('* Has _not been_ preprocessed\n')
            end            
        end

        function out = isempty(self)
            % Determines if the block is empty
            out = isempty(self.data);
        end
        
        function plot(self, varargin)
            % SYNTAX
            %
            % plot(block)                   % plots all the data in 2 x 4 subplots, or fewer
            % plot(block, {'sub', [2, 5]})  % plots all the data in 2 x 5 subplots
            % plot(block, {'one', <column name or number>})
            % plot(block, {'mark', <row name(s) or number(s)>})
            tags = 1:self.K;
            if self.K == 1
                subplot_size = [1, 1];
            elseif self.K == 2
                subplot_size = [1, 2];
            elseif self.K == 3
                subplot_size = [1, 3];
            elseif self.K == 4
                subplot_size = [2, 2];
            elseif self.K == 5
                subplot_size = [2, 3];
            elseif self.K == 6
                subplot_size = [2, 3];
            elseif self.K == 7
                subplot_size = [2, 3];
            else
                subplot_size = [2, 4];
            end
                
            
            mark = [];
            for i = 1:numel(varargin)
                key = varargin{i}{1};
                value = varargin{i}{2};
                if strcmpi(key, 'sub')
                    subplot_size = value;
                elseif strcmpi(key, 'one')
                    subplot_size = [1, 1];
                    tags = self.get_vector(2, value);
                elseif strcmpi(key, 'mark')
                    mark = self.get_vector(1, value);
                end 
            end
            
            plot_tags(self, tags, subplot_size, mark)
        end

%         function self = preprocess(self, varargin)
%             % Calculates the preprocessing vectors for a block
%             %
%             % Currently the only preprocessing model supported is to mean center
%             % and scale.
%             
%             % We'd like to preprocess the ``other`` block using settings 
%             % from the current block.
%             if nargin==2 && self.is_preprocessed
%                 other = varargin{1};
%                 if ~isa(other, 'block')
%                     error('The new data must be a ``block`` instance.');
%                 end
%                 % Don't worry about preprocessing empty blocks.
%                 if ~isempty(other)
%                     mean_center = self.PP.mean_center;
%                     scaling = self.PP.scaling;
%                     if ~other.is_preprocessed                    
%                         other.data = other.data - repmat(mean_center, other.N, 1);
%                         other.data = other.data .* repmat(scaling, other.N, 1);
%                     end
%                 else
%                     % Catches the case when empty blocks are preprocessed
%                     scaling = NaN;
%                 end
%                 other.is_preprocessed = true;
%                 
%                 % Will scaling introduce missing values?
%                 if any(isnan(scaling))
%                     other.has_missing = true;
%                 end
%                 self = other;
%                 return
%             end
%             
%             % Apply the preprocesing to the training data
%             if not(self.is_preprocessed)
%                 
%             
%                 % Centering based on the mean
%                 mean_center = nanmean(self.data, 1);
%                 scaling = nanstd(self.data, 1);
% 
%                 % Replace zero entries with NaN: this is handled later on with scaling
%                 % This will create missing data, so we need to set the flag
%                 % correctly
%                 if any(scaling < sqrt(eps))
%                     scaling(scaling < sqrt(eps)) = NaN;
%                     self.has_missing = true;
%                 end                
%                 scaling = 1./scaling;
%             
%             
%                 self.data = self.data - repmat(mean_center, self.N, 1);
%                 self.data = self.data .* repmat(scaling, self.N, 1);
% 
%                 % Store the preprocessing vectors for later on
%                 self.PP.mean_center = mean_center;
%                 self.PP.scaling = scaling;  
% 
%                 self.is_preprocessed = true;
%             end
%         end
%         
%         function self = un_preprocess(self, varargin)
%             % UNdoes preprocessing for a block.
%             %
%             % Currently the only preprocessing model supported is to mean center
%             % and scale.
%             
%             % We'd like to preprocess the ``other`` block using settings 
%             % from the current block.
%             if nargin==2 && self.is_preprocessed
%                 other = varargin{1};
%                 if ~isa(other, 'block')
%                     error('The new data must be a ``block`` instance.');
%                 end
%                 % Don't worry about preprocessing empty blocks.
%                 if ~isempty(other)
%                     mean_center = self.PP.mean_center;
%                     scaling = self.PP.scaling;
%                     if ~other.is_preprocessed                    
%                         other.data = other.data - repmat(mean_center, other.N, 1);
%                         other.data = other.data .* repmat(scaling, other.N, 1);
%                     end
%                 end                
%                 other.is_preprocessed = true;
%                 
%                 % Will scaling introduce missing values?
%                 if any(isnan(scaling))
%                     other.has_missing = true;
%                 end
%                 self = other;
%                 return
%             end
%             
%             % Apply the preprocesing to the training data
%             if not(self.is_preprocessed)
%                 
%             
%                 % Centering based on the mean
%                 mean_center = nanmean(self.data, 1);
%                 scaling = nanstd(self.data, 1);
% 
%                 % Replace zero entries with NaN: this is handled later on with scaling
%                 % This will create missing data, so we need to set the flag
%                 % correctly
%                 if any(scaling < sqrt(eps))
%                     scaling(scaling < sqrt(eps)) = NaN;
%                     self.has_missing = true;
%                 end                
%                 scaling = 1./scaling;
%             
%             
%                 self.data = self.data - repmat(mean_center, self.N, 1);
%                 self.data = self.data .* repmat(scaling, self.N, 1);
% 
%                 % Store the preprocessing vectors for later on
%                 self.PP.mean_center = mean_center;
%                 self.PP.scaling = scaling;  
% 
%                 self.is_preprocessed = true;
%             end
%         end
         
%         function [self, other] = exclude(self, dim, which)
%             % Excludes rows (``dim``=1) or columns (``dim``=2) from the block 
%             % given by entries in the vector ``which``.
%             %
%             % The excluded entries are returned as a new block in ``other``.
%             % Note that ``self`` is actually returned as a new block also,
%             % to accomodate the fact that modelling elements might have been 
%             % removed.  E.g. if user excluded rows, then the scores are not
%             % valid anymore.
%             %
%             % Example: [batch_X, test_X] = batch_X.exclude(1, 41); % removes batch 41
%             %
%             % NOTE: at this time, you cannot exclude a variable from a batch
%             % block.  To do that, exclude the variable in the raw data, before
%             % creating the block.
%             
%             
%             s_ordinary = struct; 
%             s_ordinary.type = '()';
%             s_ordinary_remain = struct; 
%             s_ordinary_remain.type = '()';
%             self_tagnames = self.tagnames;
%             if dim == 1 
%                 if any(which>self.N)
%                     error('block:exclude', 'Entries to exclude exceed the size (row size) of the block.')
%                 end
%                 s_ordinary.subs = {which, ':'};
%                 remain = 1:self.N;
%                 remain(which) = [];
%                 s_ordinary_remain.subs = {remain, ':'};
%                 other_tagnames = self.tagnames;
%             end
%             if dim == 2 
%                 if any(which>self.K)
%                     error('block:exclude', 'Entries to exclude exceed the size (columns) of the block.')
%                 end
%                 if strcmp(self.block_type, 'batch')
%                     error('block:exclude', 'Excluding tags from batch blocks is not currently supported.')
%                 end
%                 s_ordinary.subs = {':', which};
%                 
%                 other_tagnames = self.tagnames(which);
%                 self_tagnames(which) = [];
%                 remain = 1:self.K;
%                 remain(which) = [];
%                 s_ordinary_remain.subs = {':', remain};
%             end
%             
%             if strcmp(self.block_type, 'batch')
%                 other = block(subsref(self.data, s_ordinary), self.name);
%                 other.block_type = 'batch';
%                 other.nTags = self.nTags;
%                 other.J = self.J;
%                 other.tagnames = self.tagnames;
%                 other.raw_data = cell(numel(which), 1);
%                 for n = 1:numel(which)
%                     other.raw_data{n} = self.raw_data{which(n)};
%                 end
%                 
%                 self_raw_data = self.raw_data;
%                 self_data = subsref(self.data, s_ordinary_remain);
%                 self = block(self_data, self.name);
%                 self.block_type = 'batch';
%                 self.J = other.J;
%                 self.nTags = other.nTags;
%                 self.tagnames = other.tagnames;
%                 self.raw_data = cell(numel(remain), 1);
%                 for n = 1:numel(remain)
%                     self.raw_data{n} = self_raw_data{remain(n)};
%                 end
%             end
%         end
        
 
    end % end methods
end % end classdef
            
%-------- Helper functions. May NOT modify ``self``.
function plot_tags(self, tags, subplot_size, mark)
    K = size(self.data(:, tags),2);
    hA = zeros(K, 1);
    count = -prod(subplot_size);
    for k = 1:K
        if mod(k-1, prod(subplot_size))==0
            figure('Color', 'White');
            count = count + prod(subplot_size);
        end
        hA(k) = subplot(subplot_size(1), subplot_size(2), k-count);
    end
    if numel(self.labels)
        tagnames = self.labels{2,1};
    end
            
    for k = 1:K
        plot(hA(k), self.data(:,tags(k)), 'k')
        title(hA(k), tagnames{k}, 'FontSize',14)
        set(hA(k), 'FontSize',14)
        axis tight
        grid(hA(k),'on')
    end
        
end

% function plot_loadings(self, which_loadings)  
%         
%     if strcmpi(self.block_type, 'batch')
% 
%         nSamples = self.J;                                                              % Number of samples per tag
%         nTags = self.nTags;                                                                  % Number of tags in the batch data
%         tagNames = char(self.tagnames);
%         
%         for a = which_loadings
%             data = self.P(:, a);
%             y_axis_label = ['Loadings, p_', num2str(a)];
%             
%             data = reshape(data, self.nTags, self.J)';
%             cum_area = sum(abs(data));
%             data = data(:);
%             hF = figure('Color', 'White');
%             hA = axes;
%             bar(data);
% 
%             x_r = xlim;
%             y_r = ylim;
%             xlim([x_r(1,1) nSamples*self.nTags]);
%             tick = zeros(self.nTags,1);
%             for k=1:self.nTags
%                 tick(k) = nSamples*k;
%             end
% 
%             for k=1:self.nTags
%                 text(round((k-1)*nSamples+round(nSamples/2)), ...
%                      diff(y_r)*0.9 + y_r(1),deblank(tagNames(k,:)), ...
%                      'FontWeight','bold','HorizontalAlignment','center');
%                 text(round((k-1)*nSamples+round(nSamples/2)), ...
%                      diff(y_r)*0.05 + y_r(1), sprintf('%.2f',cum_area(k)), ...
%                      'FontWeight','bold','HorizontalAlignment','center');
%             end
% 
%             set(hA,'XTick',tick);
%             set(hA,'XTickLabel',[]);
%             set(hA,'Xgrid','On');
%             xlabel('Batch time repeated for each variable');
%             ylabel(y_axis_label);
%             pos0 = get(0,'ScreenSize');
%             delta = pos0(3)/100*2;
%             posF = get(hF,'Position');
%             set(hF,'Position',[delta posF(2) pos0(3)-delta*2 posF(4)]);
%         end
%         
%         
%     end
% end
