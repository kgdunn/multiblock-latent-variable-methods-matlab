% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% Licensed under the BSD license.
% -------------------------------------------------------------------------
%
% Code for creating data blocks. A block is the smallest subset of data in a
% model.

classdef block_base < handle
    
    % Class attributes
    properties
        data = false;           % Block's data
        has_missing = false;    % {true, false}
        mmap = [];              % Missing data map
        name = '';              % Name of the block
        name_type = 'auto';     % {'auto', 'given'}
        labels = {};            % Cell array: rows are the modes; columns are sets of labels
                                % 2 x 3: row labels and columns labels, with
                                % up to 3 sets of labels for each dimension
                                
        N = 0;                  % Number of observations (rows)
        K = 0;                  % Number of variables (columns)
        source = '';            % Where the data came from
    end
    
    methods
        function self = block_base(given_data, block_name, source, varargin)
            % SYNTAX:
            % block_base(data, block_name, variable arguments in cell arrays)
            
            % Store whatever data was given us
            self.data = given_data;
            
            self.labels = cell(ndims(given_data), 0);    % 0-columns of labels
            
            % Missing data handling
            self.mmap = [];             
            missing_map = ~isnan(given_data);   % 0=missing, 1=present
            if not(all(missing_map(:)))
                self.mmap = missing_map;
            end
            
            % Second argument: block name
            if not(isempty(block_name))
                self.name = block_name;
                self.name_type = 'given';
            end
            
            % Third argument: block source
            self.source = source;
            
            
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
                        
                    % ``source``
                    elseif strcmpi(key, 'source')
                        self.source = value;
                        
                    end
                end
            end
        end
        
        function out = get.N(self)
            out = size(self.data, 1);
        end
        
        function out = get.K(self)
            out = size(self.data, 2);
        end
        
        function out = get.has_missing(self)
            out = true;           
            if isempty(self.mmap)
                out = false;
            end            
        end
        
        function out = get.name(self)
            if strcmp(self.name_type, 'auto')
                out = get_auto_name(self);
            else
                out = self.name;
            end
        end
        
        function out = get_auto_name(self)
            out = ['block-', num2str(self.N), '-', num2str(self.K)];
        end
        
        function out = get_data(self)
            % Returns the data array stored in self.
            out = self.data;
        end
        
        function out = shape(self, varargin)
            out = size(self.data);
            if nargin > 1
                out = out(varargin{1});
            end
        end
        
        function size(self) %#ok<MANU>
            error('block_base:size', 'Use the ``shape(...)`` function to obtain the shape of a block object.');
        end
        
        function out = numel(self)
            out = numel(self.data);
        end
        
        function out = copy(self)
            props = properties(self);
            
            out = self.new();
            for i=1:numel(props)
                out.(props{i}) = self.(props{i});
            end
        end
        
        function y = mean(self, varargin)
            y = block_base.nanmean(self.data, varargin{:});
        end
        
        function y = std(self, dim, flag)
            % FORMAT: Y = NANSTD(X,DIM,FLAG)
            % 
            %    Standard deviation ignoring NaNs
            %
            %    This function enhances the functionality of NANSTD as distributed in
            %    the MATLAB Statistics Toolbox and is meant as a replacement (hence the
            %    identical name).  
            %
            %    NANSTD(X,DIM) calculates the standard deviation along any dimension of
            %    the N-D array X ignoring NaNs.  
            %
            %    NANSTD(X,DIM,0) normalizes by (N-1) where N is SIZE(X,DIM).  This make
            %    NANSTD(X,DIM).^2 the best unbiased estimate of the variance if X is
            %    a sample of a normal distribution. If omitted FLAG is set to zero.
            %    
            %    NANSTD(X,DIM,1) normalizes by N and produces the square root of the
            %    second moment of the sample about the mean.
            %
            %    If DIM is omitted NANSTD calculates the standard deviation along first
            %    non-singleton dimension of X.
            %
            %    Similar replacements exist for NANMEAN, NANMEDIAN, NANMIN, NANMAX, and
            %    NANSUM which are all part of the NaN-suite.
            %
            %    See also STD

            % -------------------------------------------------------------------------
            %    author:      Jan Gläscher
            %    affiliation: Neuroimage Nord, University of Hamburg, Germany
            %    email:       glaescher@uke.uni-hamburg.de
            %    
            %    $Revision: 1.1 $ $Date: 2004/07/15 22:42:15 $
            
            x = self.data;
            
            if isempty(x)
                y = NaN;
                return
            end

            if nargin < 3
                flag = 0;
            end

            if nargin < 2
                dim = find(size(x)~=1, 1);
                if isempty(dim)
                    dim = 1; 
                end	  
            end

            % Find NaNs in x and nanmean(x)
            nans = isnan(x);
            avg = block_base.nanmean(x,dim);

            % create array indicating number of element 
            % of x in dimension DIM (needed for subtraction of mean)
            tile = ones(1,max(ndims(x),dim));
            tile(dim) = size(x,dim);

            % remove mean
            x = x - repmat(avg,tile);

            count = size(x,dim) - sum(nans,dim);

            % Replace NaNs with zeros.
            x(isnan(x)) = 0; 

            % Protect against a  all NaNs in one dimension
            i = find(count==0);

            if flag == 0
                y = sqrt(sum(x.*x,dim)./max(count-1,1));
            else
                y = sqrt(sum(x.*x,dim)./max(count,1));
            end
            y(i) = i + NaN;
            % $Id: nanstd.m,v 1.1 2004/07/15 22:42:15 glaescher Exp glaescher $
        end
        
        function y = var(self, varargin)
            y = self.std(varargin{:}) .^ 2;
        end
            
        function y = sum(self, dim)
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
            x = self.data;
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
            % $Id: nansum.m,v 1.2 2005/06/13 12:14:38 glaescher Exp glaescher $
        end
        
        function add_labels(self, dim, to_add)
            added = false;
            try
                to_add = cellstr(to_add);
            catch ME
                if iscell(to_add)
                    to_add = cellstr(num2str(cell2mat(to_add(:))));
                else
                    to_add = cellstr(num2str(to_add(:)));
                end
            end
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
        
        function [idx, idx_names] = index_by_name(self, axis, names)
            % Gets the index/indicies into an ``axis`` from the label names
            % provided in ``names``.
            idx = [];
            idx_names = {};
            ax_labels = self.labels(axis,1);
            ax_labels = ax_labels{1};
            
            if ischar(names)
                names = {names};
            end
            
            % Perhaps the ``names`` are just a vector of indices
            if isempty(ax_labels)
                ax_dim = self.shape(axis);
                for j = 1:numel(names)
                    if names(j) <= ax_dim
                        idx(end+1) = names(j);
                        idx_names{end+1} = num2str(names(j));
                    end
                end
                return
            end
            
            for j = 1:numel(names)
                name_j = names(j);
                for k = 1:numel(ax_labels)
                    if strcmp(name_j, strtrim(ax_labels(k)))
                        idx(end+1) = k;
                        idx_names(end+1) = ax_labels(k);
%                    end
                    % Deal with strings of numbers
                    elseif str2double(ax_labels(k)) == str2double(name_j)
                        idx(end+1) = k;
                        idx_names(end+1) = ax_labels(k);
                    end                        
                end                
            end
        end

        function out = isempty(self)
            % Determines if the block is empty
            out = isempty(self.data);
        end
        
        function disp(self)
            % Displays a text summary of the block
            fprintf('%s: %d observations and %d variables\n', self.name, self.N, self.K)
            if self.has_missing
                fprintf('* Has missing data\n')
            else
                fprintf('* Has _no_ missing data\n')
            end
            
        end
        
        function varargout = plot(self, varargin)
            % SYNTAX
            %
            % plot(block)                      % plots all the data in 2 x 5 subplots, or fewer
            % plot(block, {'layout', [2, 3]})  % plots all the data in 2 x 3 subplots
            % plot(block, {'one', <column name or number>})
            % plot(block, {'mark', <row name(s) or number(s)>})
            
            % Set the defaults
            default_layout = [2, 5];
            subplot_size = block_base.optimal_layout(self.K, default_layout);
            tags = 1:self.K;
            mark = NaN;
            footer_string = {datestr(now)};
            
            for i = 1:numel(varargin)
                key = varargin{i}{1};
                value = varargin{i}{2};
                if strcmpi(key, 'layout')
                    subplot_size = value;
                elseif strcmpi(key, 'one')
                    subplot_size = [1, 1];
                    tags = self.index_by_name(2, value);
                elseif strcmpi(key, 'mark')
                    mark = self.index_by_name(1, value);
                end 
            end
            
            [h, hHeaders, hFooters, title_str] = plot_tags(self, tags, subplot_size, mark);
            self.add_plot_footers(hFooters, footer_string);
            self.add_plot_window_title(hHeaders, title_str)
            for i=1:nargout
                varargout{i} = h;
            end
        end
               
        function add_plot_footers(self, hFooters, footer_string)
            % Convert the cell string to a long char string
            foot = footer_string{1};
            for j = 2:numel(footer_string)
                foot = [foot, footer_string{j}];
            end
            
            % Append the source file to the figure
            footer_string = [foot, ' [source: ', self.source, ']'];
            for k = 1:numel(hFooters)
                set(hFooters(k), 'String', footer_string)
            end
        end
        
        function [block_data, varargout] = preprocess(self, varargin)
            % Calculates the preprocessing vectors for a block, ``pp`` and 
            % returns the preprocessed ``data``, computed on ``self``, 
            % but does not modify ``self``.
            %
            % Currently the only preprocessing model supported is to mean center
            % and scale.
            %
            % SYNTAX:
            %
            % Preprocesses ``self`` and returns the PP structure and the PP data.
            %
            % [data, pp] = self.preprocess()
            %
            % Returns the ``other_block`` in preprocessed form, from the given
            % preprocessing structure, ``pp``:
            %
            % other_pp = self.preprocess(other_block, pp)
            
            % We'd like to preprocess the ``other`` block using settings 
            % from the current block.
            if nargin==3 
                other = varargin{1};
                PP = varargin{2};
                if ~isa(other, 'block_base')
                    error('The new data must be a ``block`` instance.');
                end
                % Don't worry about preprocessing empty blocks.
                if ~isempty(other)
                    mean_center = PP.mean_center;
                    scaling = PP.scaling;
                    block_data = other.get_data() - repmat(mean_center, other.N, 1);
                    block_data = block_data .* repmat(scaling, other.N, 1);
                else
                    % Catches the case when empty blocks are preprocessed
                    scaling = NaN;
                end
                
                % Will scaling introduce missing values?
                if any(isnan(scaling))
                    % TODO(KGD): how to do this programmatically?
                    % in case we change mmap definition in the future?
                    other.mmap = ~isnan(scaling);
                end
                
                other.data = block_data;
                block_data = other;
                return
            end
            
            % Calculate the preprocesing information from the training data
            block_data = self.copy();
            
            % Centering based on the mean
            mean_center = mean(block_data, 1);
            scaling = std(block_data, 1);

            % Replace zero entries with NaN: this is handled later on with scaling
            % This will create missing data, so we need to set the flag
            % correctly
            if any(scaling < sqrt(eps))
                scaling(scaling < sqrt(eps)) = NaN;
                self.mmap = ~isnan(scaling);
                block_data.mmap = ~isnan(scaling);
            end                
            scaling = 1./scaling;

            block_data.data = block_data.data - repmat(mean_center, self.N, 1);
            block_data.data = block_data.data .* repmat(scaling, self.N, 1);

            % Return the preprocessing vectors
            if nargout > 1
                PP = struct;
                PP.mean_center = mean_center;
                PP.scaling = scaling;
                varargout{1} = PP;
            end
        end

        function out = ssq(self, varargin)
            out = ssq(self.data, varargin{:});
        end
        
        function out = un_preprocess(self, data, pp)
            % UNdoes preprocessing for a block.
            %
            % Currently the only preprocessing model supported is to mean center
            % and scale.
            if isempty(data)
                data = self.data;
            end
            out = data ./ repmat(pp.scaling, self.N, 1);
           	out = out + repmat(pp.mean_center, self.N, 1);
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

            
            if not(isnumeric(which))
                error('block_batch:exclude', 'Exclude based on numeric indices only, not based on row or column labels.')
            end

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
                if any(which>self.K)
                    error('block:exclude', 'Entries to exclude exceed the size (columns) of the block.')
                end            
                exc_s.subs = {':', which};
                remain_idx = 1:self.K;
                remain_idx(which) = [];
                rem_s.subs = {':', remain_idx};
            end

            other = self.copy();

            self.data = subsref(self.data, rem_s);
            other.data = subsref(other.data, exc_s);
            if numel(self.mmap) > 1
                self.mmap = subsref(self.mmap, rem_s);
                other.mmap = subsref(other.mmap, exc_s);
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

     end % end methods (ordinary)
    
    % Subclasses may not redefine these methods
    methods (Sealed=true)
    end % end methods (sealed)
    
    % These methods don't invoke ``self``
    methods (Static=true)
        function out = new()
            % Create a new copy of self with no data.
            out = block_base([], [], '');
        end
        
        function subplot_size = optimal_layout(nTags, default_layout, override)
            % This function can be improved so that the optimal layout "builds
            % up" to the default_layout.  E.g what if default_layout is [3,6]?
            
            if nargin>2 && not(any(isnan(override(:))))
                subplot_size = override;
                return;
            end
            
            if nTags == 1
                subplot_size = [1, 1];
            elseif nTags == 2
                subplot_size = [1, 2];
            elseif nTags == 3
                subplot_size = [1, 3];
            elseif nTags == 4
                subplot_size = [2, 2];
            elseif nTags == 5
                subplot_size = [2, 3];
            elseif nTags == 6
                subplot_size = [2, 3];
            elseif nTags == 7
                subplot_size = [2, 3];
            else
                subplot_size = default_layout;
            end
        end
        
        function [hF, hHead, hFoot] = add_figure()
            % Adds a new figure
            background_colour = [1, 1, 1];
            font_size = 14;
            hF = figure('Color', background_colour);
            set(hF, 'ToolBar', 'figure')
            units = get(hF, 'Units');
            set(hF, 'units', 'Pixels');
             
            screen = get(0,'ScreenSize');   
%             fPos = get(hF, 'position');
%             fPos(1) = round(0.10*screen(3));
%             fPos(2) = round(0); %.10*screen(4));
%             fPos(3) = screen(3)-2*screen(1);
%             fPos(4) = screen(4)-2*screen(2);
%             set(hF, 'Position', fPos);
%             
            fPos = get(hF, 'position');
            fPos(1) = round(0.05*screen(3));
            fPos(2) = round(0.1*screen(4));
            fPos(3) = 0.90*screen(3);
            fPos(4) = 0.80*screen(4);
            set(hF, 'Position', fPos);
            
            
  
            Txt.Interruptible = 'off';  % Text fields different to default values
            Txt.BusyAction    = 'queue';
            Txt.Style         = 'text';
            Txt.Units         = 'normalized';
            Txt.HorizontalAli = 'center';
            Txt.Background    = background_colour;
            Txt.FontSize      = font_size;
            Txt.Parent        = hF;
            set(hF, 'Units', 'normalized')
            
            hHead = uicontrol(Txt, ...
                             'Position',[0, 0, eps, eps], ...  %'Position',[0.02, 0.95, 0.96 0.04], ...
                             'ForegroundColor',[0 0 0], 'String', '');
            %set(hHead, 'Units', 'Pixels')
            %p = get(hHead, 'Position')
            Txt.FontSize = 10;
            hFoot = uicontrol(Txt, ...
                             'Position',[0.02, 0.005, 0.96 0.04], ...
                             'ForegroundColor',[0 0 0], 'String', '', ...
                             'HorizontalAlignment', 'left');
            set(hF, 'units', units);
        end
        
        function add_plot_window_title(hHeaders, text_to_add)
            for k = 1:numel(hHeaders)
                hF = get(hHeaders(k), 'Parent');
                set(hF, 'Name', text_to_add);
            end
        end
        
        function y = nanmean(x, dim)
            % FORMAT: Y = NANMEAN(X,DIM)
            % 
            %    Average or mean value ignoring NaNs
            %
            %    This function enhances the functionality of NANMEAN as distributed in
            %    the MATLAB Statistics Toolbox and is meant as a replacement (hence the
            %    identical name).  
            %
            %    NANMEAN(X,DIM) calculates the mean along any dimension of the N-D
            %    array X ignoring NaNs.  If DIM is omitted NANMEAN averages along the
            %    first non-singleton dimension of X.
            %
            %    Similar replacements exist for NANSTD, NANMEDIAN, NANMIN, NANMAX, and
            %    NANSUM which are all part of the NaN-suite.
            %
            %    See also MEAN

            % -------------------------------------------------------------------------
            %    author:      Jan Gläscher
            %    affiliation: Neuroimage Nord, University of Hamburg, Germany
            %    email:       glaescher@uke.uni-hamburg.de
            %    
            %    $Revision: 1.1 $ $Date: 2004/07/15 22:42:13 $
            if isempty(x)
                y = NaN;
                return
            end

            if nargin < 2
                dim = find(size(x)~=1, 1 );
                if isempty(dim)
                    dim = 1;
                end
            end

            % Replace NaNs with zeros.
            nans = isnan(x);
            x(isnan(x)) = 0; 

            % denominator
            count = size(x,dim) - sum(nans,dim);

            % Protect against a  all NaNs in one dimension
            i = find(count==0);
            count(i) = ones(size(i));

            y = sum(x,dim)./count;
            y(i) = i + NaN;
            % $Id: nanmean.m,v 1.1 2004/07/15 22:42:13 glaescher Exp glaescher $
        end
    end % end methods (static)

%     % Subclass must redefine these methods
%     methods (Abstract=true)
%         exclude_post(self);
%     end % end methods (abstract)
end % end classdef
            
%-------- Helper functions. May NOT modify ``self``.
function [hA, hHeaders, hFooters, title_str] = plot_tags(self, tags, subplot_size, mark)
    K = size(self.data(:, tags),2);
    hA = zeros(K, 1);
    hHeaders = [];
    hFooters = [];
    
    count = -prod(subplot_size);
    for k = 1:K
        if mod(k-1, prod(subplot_size))==0
            [hF, hHead, hFoot] = self.add_figure(); %#ok<ASGLU>
            hHeaders(end+1) = hHead; %#ok<*AGROW>
            hFooters(end+1) = hFoot;
            count = count + prod(subplot_size);
        end
        hA(k) = subplot(subplot_size(1), subplot_size(2), k-count);
    end
    if numel(self.labels)
        tagnames = self.labels{2,1};
    else
        tagnames = cell(1, self.K);
        for k = 1:self.K
            tagnames{k} = ['Tag ', num2str(k)];
        end
    end
    highlight_colour = [255, 102, 0]/255;
    for k = 1:K
        hPlot = plot(hA(k), self.data(:,tags(k)), 'k');
        title(hA(k), char(tagnames{k}), 'FontSize',14)
        set(hA(k), 'FontSize',14)
        axis(hA(k), 'tight')
        extent = get(hA(k), 'YLim');
        delta = diff(extent(1:2))*0.05;
        set(hA(k), 'YLim', [extent(1)-delta extent(2)+delta])
        grid(hA(k),'on')
        if not(isnan(mark))
            set(hA(k),'Nextplot', 'add')
            x_data = get(hPlot, 'XData');
            y_data = get(hPlot, 'YData');
            plot(hA(k), x_data(mark), y_data(mark), 's', 'Color', highlight_colour, ...
                'MarkerSize', 5, 'Linewidth', 2)
        end        
    end
    title_str = 'Plots of raw data';
end



% function plot_loadings(self, which_loadings)  
%         
%     if strcmpi(self.block_type, 'batch')
% 
%         nSamples = self.J;                   % Number of samples per tag
%         nTags = self.nTags;                  % Number of tags in the batch data
%         tagNames = char(self.tagnames);
%         
%         for a = which_loadings
%             data = self.P(:, a);
%             y_axis_label = ['Loadings, p_', num2str(a)];
%             
%             data = reshape(data, self.nTags, self.J)';
%             cum_area = sum(abs(data));
%             data = data(:);
%             [hF, hHead, hFoot] = add_figure();
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
