% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for creating data blocks. A block is the smallest subset of data in a
% model.

classdef block_batch < block_base
    
    % Class attributes
    properties
        % Batch-specific properties       
        nTags = 0;      % Number of batch tags        
        J = 0;          % Number of batch time steps
        batch_raw = {}; % Raw data from the batch
    end
    
    methods
        function self = block_batch(given_data, block_name, source, varargin)
            
            % Initialize the superclass, but do almost nothing
            self = self@block_base([], [], source);
            
            % Batch data blocks are always 3-dimensional
            self.labels = cell(3, 0); 
            
            nBatches = -1;
            
            % Third and subsequent arguments: 
            if nargin > 2
                for i = 1:numel(varargin)
                    key = varargin{i}{1};
                    value = varargin{i}{2};
                    
                    if strcmpi(key, '__shell__')
                        return
                    
                    elseif strcmpi(key, 'nbatches')
                        nBatches = max(nBatches, floor(value));     
                    
                    elseif strcmpi(key, 'batch_names')
                        self.add_labels(1, value);
                        nBatches = max(nBatches, numel(value));
                        
                    elseif strcmpi(key, 'col_labels')
                        self.add_labels(2, value);
                        
                    elseif strcmpi(key, 'batch_tag_names')
                        self.add_labels(2, value);
                        
                    elseif strcmpi(key, 'time_names')
                        self.add_labels(3, value);
                        
                    elseif strcmpi(key, 'source')
                        self.source = value;
                        
                    end
                end
            end
            if nBatches < 0
                error('block_batch:number_of_batches_not_specified', ...
                     ['Please specify the number of batches. For example: ', ...
                      ' {"nbatches", 12} '])
            end
            
            N_init = size(given_data, 1);
            K_init = size(given_data, 2);
                               
            % Reshape the data
            self.J = N_init / nBatches;
            self.nTags = K_init;
            if self.J ~= floor(self.J)
                error('block_batch:inconsistent_data_specification',['There are %d rows and you ', ...
                    'specified %d batches.  This is not consistent with the data provided.'], ...
                    N_init, nBatches)
            end
            
            % Unfold the data from standard form into latent variable form            
            [self.batch_raw, self.data] = self.standard_form_to_latent_variable_form(given_data, nBatches);
            
                        
            % Missing data handling
            self.mmap = false;            
            missing_map = ~isnan(self.data);   % 0=missing, 1=present
            if not(all(missing_map))
                self.mmap = missing_map;
            end
            
            % Block name
            if not(isempty(block_name))
                self.name = block_name;
                self.name_type = 'given';
            end
            
        end
        
        function [cell_form, data_form] = standard_form_to_latent_variable_form(self, given_data, nBatches)
            % Unfold the data from standard form into latent variable form
            % Unfolded order is to have all tags at time 1, then all the 
            % tags at time 2, and so on, up till time step J.
            % So we write that X = K*J (K tags, at J time steps).
            
            % The ``cell_form`` is convenient for batch plotting and excluding
            % batches later on.  The ``data_form`` is for LVM.
            cell_form =  cell(nBatches,1);
            data_form = zeros(nBatches, self.J * self.nTags);
            for n = 1:nBatches
                startRow = (n-1) * self.J+1;
                endRow = startRow - 1 + self.J;
                cell_form{n} = given_data(startRow:endRow,:);
                temp = cell_form{n}';
                data_form(n, :) = temp(:)';
            end
        end
        
        function out = get_auto_name(self)
            out = ['batch-', num2str(self.N), '-', num2str(self.K), '-', num2str(self.J)];
        end
                
        function out = get_data(self)
            % Returns the data array stored in self.
            out = self.data;
        end
        
        function disp_header(self)
            % Displays a text summary of the block
            fprintf('%s: %d batches, %d variables, %d time samples (batch unfolded)\n', self.name, self.N, self.nTags, self.J)
        end
        
        function out = shape(self, varargin)
            out = zeros(1, 3);
            out(1) = self.N;
            out(2) = self.nTags;
            out(3) = self.J;
            
            if nargin > 1
                out = out(varargin{1});
            end
        end
        
        function [self, other] = exclude(self, dim, which)
            % Excludes rows (``dim``=1) or columns (``dim``=2) from the block 
            % given by entries in the vector ``which``.
            %
            % The excluded entries are returned as a new block in ``other``.
            % ``other`` will retain all properties originally in ``self``.
            %
            % Example: [batch_X, test_X] = batch_X.exclude(1, 41); % removes batch 41
            %
            % NOTE: at this time, you cannot exclude a variable from a batch
            % block.  To do that, exclude the variable in the raw data, before
            % creating the block.


            exc_s = struct; 
            exc_s.type = '()';

            rem_s = struct; 
            rem_s.type = '()';

            if dim == 1 
                if any(which>self.N)
                    error('block:exclude', 'Entries to exclude exceed the size (row size) of the block.')
                end
                exc_s.subs = {which, ':'};
                remain_idx = 1:self.N;
                remain_idx(which) = [];
                rem_s.subs = {remain_idx, ':'};
            end
            if dim == 2 
                if any(which>self.nTags)
                    error('block:exclude', 'Entries to exclude exceed the size (columns) of the block.')
                end            
                exc_s.subs = {':', which};
                remain_idx = 1:self.nTags;
                remain_idx(which) = [];
                rem_s.subs = {':', remain_idx};
            end

            other = self.copy();
            if dim == 1
                self.batch_raw = subsref(self.batch_raw, rem_s);
                other.batch_raw = subsref(other.batch_raw, exc_s);            
                self.data = subsref(self.data, rem_s);
                other.data = subsref(other.data, exc_s);
            end
            if dim == 2
                temp = (subsref(self.batch_raw{1}, rem_s))';
                temp = temp(:)';
                new_K_self = numel(temp);
                
                temp = (subsref(other.batch_raw{1}, exc_s))';
                temp = temp(:)';
                new_K_other = numel(temp);
                    
                nBatches = self.N;
                self.data = zeros(nBatches, new_K_self) .* NaN;
                other.data = zeros(nBatches, new_K_other) .* NaN;
                for n = 1:nBatches
                    self.batch_raw{n} = subsref(self.batch_raw{n}, rem_s);
                    temp = self.batch_raw{n}';
                    self.data(n,:) = temp(:)';
                    
                    other.batch_raw{n} = subsref(other.batch_raw{n}, exc_s);     
                    temp = other.batch_raw{n}';
                    other.data(n,:) = temp(:)';
                end
                
                self.nTags = numel(remain_idx);
                other.nTags = numel(which);
            end
            
            tagnames = self.labels(dim,:);
            exc_tag = struct;
            exc_tag.type = '()';
            exc_tag.subs = {which};
            rem_tag = struct;
            rem_tag.type = '()';
            rem_tag.subs = {remain_idx};
            for entry = 1:numel(tagnames)
                tags = tagnames{entry};
                if not(isempty(tags))
                    self.labels{dim, entry} = subsref(tags, rem_tag);
                    other.labels{dim, entry} = subsref(tags, exc_tag);
                end
            end        
        end
        
        function varargout = plot(self, varargin)
            % X = randn(50,4);
            % batch_names = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'Ten'};
            % time_names = {'1', '2', '3', '4', '5'};
            % tag_names = {'Temp', 'Pres', 'Flow', 'Heat input', 'Flow2'};
            % b = block(X, 'X block', {'batch_names', batch_names}, ...
            %                         {'batch_tag_names', tag_names}, ....
            %                         {'time_names', time_names});
            % 
            % % Basic batch block plot: shows each trajectory for all batches
            % h = plot(b);
            %
            % % Specify the layout for the trajectories: 2x5 subplots per window
            % h = plot(b, {'layout', [2, 5]});
            % 
            % % Specify a batch or batches to highlight
            % h = plot(b, {'mark', 'X18'});
            % h = plot(b, {'mark', [42, 43]});
            % 
            % % Plot only a certain batch (TODO)
            % h = plot(b, {'some', 'A'});
            % h = plot(b, {'some', ['A', 'B']});
            % 
            % % Show the preprocessed batch data instead (the default is to show the raw
            % % data)
            % h = plot(b, {'pp', true});

            % Set the defaults
            default_layout = [2, 5];
            layout_override = NaN;
            layout = block_batch.optimal_layout(self.nTags, default_layout, layout_override);
            which_tags = 1:self.nTags;
            mark = NaN;
            show_preprocessed = false;
            footer_string = {datestr(now)};
            
            % Process the options
            for i = 1:numel(varargin)
                key = varargin{i}{1};
                value = varargin{i}{2};
                
                if strcmpi(key, 'layout')
                    layout = block_batch.optimal_layout(self.nTags, default_layout, value);
                    
                elseif strcmpi(key, 'mark')
                    [mark, mark_names] = self.index_by_name(1, value);
                    if numel(mark)>0
                        footer_names = mark_names{1};
                        for n = 2:numel(mark_names)
                            footer_names = [footer_names, ', ', mark_names{n}];
                        end
                        if numel(mark) > 1                            
                            footer_string{end+1} = ['; marked batches: ', footer_names]; %#ok<*AGROW>
                        elseif numel(mark) == 1
                            footer_string{end+1} = ['; marked batch: ', footer_names]; %#ok<*AGROW>
                        end
                    end
                elseif strcmpi(key, 'some')
                    which_tags = self.index_by_name(1, value);
                    layout = block_batch.optimal_layout(numel(which_tags), default_layout, layout_override);
                    
                elseif strcmpi(key, 'pp')
                    show_preprocessed = value;
                    footer_string{end+1} = '; preprocessed data';
                end 
            end
            
            if show_preprocessed
                error('TODO still')
            end
            
                           
            [hA, hHeaders, hFooters] = plot_raw(self, layout, which_tags, mark);
            
            self.add_plot_footers(hFooters, footer_string);
            
            self.add_plot_window_title(hHeaders, 'Plots of batch data');
            
            for i=1:nargout
                varargout{i} = hA;
            end
                         

        end
    end % end methods
    
    methods (Static=true)
        function out = new()
            % Create a new copy of self with no data.
            out = block_batch([], [], '', {'__shell__', true});
        end
    end
    
end % end classdef
            
%-------- Helper functions. May NOT modify ``self``.
function [hA, hHeaders, hFooters] = plot_raw(self, layout, which_tags, mark)
    % Plots the raw batch data
    % * layout:        2x1 vector, telling the desired subplot layout
    % * which_tags:    a vector of indicies into 1:self.nTags

    plot_colour = [0.1, 0.1, 0.1];
    
    highlight_colour = [255, 102, 0]/255;
    
    hA = zeros(numel(which_tags), 1);
    hHeaders = [];
    hFooters = [];
    n_plots = prod(layout);
    count = -n_plots;
    
   
    for k = 1:numel(which_tags)
        if mod(k-1, n_plots)==0
            [hF, hHead, hFoot] = self.add_figure(); %#ok<ASGLU>
            hHeaders(end+1) = hHead;
            hFooters(end+1) = hFoot;
            count = count + n_plots;
        end
        hA(k) = subplot(layout(1), layout(2), k-count);
    end

    for k = 1:numel(which_tags)
        tag = which_tags(k);
        axes(hA(k)) %#ok<LAXES>
        hold on
        for n = 1:self.N            
            plot(self.batch_raw{n}(:,tag), 'Color', plot_colour)
        end
        % Add the highlights afterwards
        for n = 1:self.N            
            if find(mark == n)
                plot(self.batch_raw{n}(:,tag), 'Color', highlight_colour, 'Linewidth', 1.5)
            end
        end
        
        set(hA(k),'FontSize',14)
        axis tight
        grid('on')
        a=axis;
        r = a(4) - a(3);
        axis([0.5 self.J+1 a(3)-0.1*r a(4)+0.1*r]);
        title(self.labels{2}{k})
    end
end


% function plot_one_batch(self,  which_batch, which_tags)
%     figure('Color', 'White');
%     hA = axes;
%     maxrow = zeros(1, self.nTags)*-Inf;
%     minrow = zeros(1, self.nTags)*Inf;
%     for n = 1:self.N            
%         maxrow = max(maxrow, max(self.raw_data{n}));
%         minrow = min(minrow, min(self.raw_data{n}));
%     end
% 
% 
%     for k = which_tags
%         plot((self.raw_data{which_batch}(:,k)-minrow(k))/maxrow(k),'k', ...
%               'LineWidth',2)
%         hold on
%         set(hA,'FontSize',14)
%         axis tight
%         a=axis;
%         r = a(4)-a(3);
%         axis([1 self.J+1 a(3)-0.1*r a(4)+0.1*r]);
%     end
% end
