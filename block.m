% Copyright (c) 2010 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for creating data blocks.
%
% A block is the smallest subset in a latent variable model.  It has ``N``
% rows and ``K`` columns.
% 
% This class holds both the data, and summary statistics from the data.


classdef block
    % Class attributes
    properties
        raw_data = false;       % Raw data for the block: only used for batch blocks
        data = false;           % Working data: will be processed by the software
        %data_pred = [];         % Predicted values of the data: only used in PLS models
        %data_pred_pp = [];      % Predicted values of the data, pre-processed: only used in PLS models
        
        has_missing = false;
        is_preprocessed = false;
        
        block_type = 'ordinary';
        name = '';
        name_type = 'default';
        
        % Labelling
        tagnames = {};
        
        % Internal structures
        mmap = false;        
        PP = struct;        % Preprocessing options
        stats = struct;     % Statistics for this block
        lim = struct;       % Statstical limits for the model entries
        
        % Batch data specific
%        
        nTags = 0;          % Number of batch tags
        
        % Block parameters
        A = 0;       % Number of components
        N = 0;       % Number of observations
        K = 0;       % Number of variables (columns)
        J = 0;       % Number of batch time steps (non-zero for batch blocks)
        
        
    end
    
    methods
        function self = block(varargin)
            % SYNTAX
            % block(data, block name, type of data, 'tagNames', tag names, 'nBatches', num batches)
          
            
            if nargin == 0
                given_data = [];
            else
                given_data = varargin{1};
            end
            if isa(given_data, 'block')
                self = given_data;
                return
            end            
            
            [self.N, self.K] = size(given_data);
            self.mmap = false;            
            missing_map = ~isnan(given_data);   % 0=missing, 1=present
            if all(missing_map)
                self.has_missing = false;
            else
                self.has_missing = true;
                self.mmap = missing_map;
            end
            
            % Second argument: block name
            try
                self.name = varargin{2};
                self.name_type = 'given';
            catch ME
                self.name = ['block-', num2str(self.N), '-', num2str(self.K)];
                self.name_type = 'default';
            end
            
            % Third argument: block type
            try
               self.block_type = lower(varargin{3});
            catch ME
               self.block_type = 'ordinary';
            end
            valid_types = {'ordinary', 'batch'};
            found = false;
            for t = 1:numel(valid_types)
                if strcmp(self.block_type, valid_types{t})
                    found = true;
                end
            end
            if not(found)
                message = 'Invalid block type: it should be either "ordinary" or "batch"';                
                error('block:invalid_block_type', message)
            end
            
            
            % Fourth/Fifth argument: 'tagNames', tag names
            try                
                self.tagnames = varargin{5};
            catch ME
                self.tagnames = [];
            end
            if isempty(self.tagnames)
                self.tagnames = cell(self.K, 1);
                for k = 1:self.K
                    self.tagnames{k} = ['V', num2str(k)];
                end 
            end
            
            % Sixth/Seventh argument: 'nBatches', number of batches
            try                
                nBatches = floor(varargin{7});
            catch ME
                if strcmp(self.block_type, 'batch')
                    error('block:number_of_batches_not_specified', ...
                        ['If specifying batch data, you must also specify ', ...
                        'the number of batches.'])
                end
            end
            if strcmp(self.block_type, 'batch')
                self.J = self.N / nBatches;
                self.nTags = self.K;
                if self.J ~= floor(self.J)
                    error('block:inconsistent_data_specification',['There are %d rows and you ', ...
                        'specified %d batches.  This is not consistent with the data provided.'], ...
                        self.N, nBatches)
                end
                
                % Unfold the data:
                % Unfolded order is to have all tags at time 1, then all the 
                % tags at time 2, and so on, up till time step J.
                % So we write that X = K*J (K tags, at J time steps).
                
                self.raw_data = cell(nBatches,1);
                self.data = zeros(nBatches, self.J * self.nTags);
                for n = 1:nBatches
                    startRow = (n-1) * self.J+1;
                    endRow = startRow - 1 + self.J;
                    self.raw_data{n} = given_data(startRow:endRow,:);
                    temp = self.raw_data{n}';
                    self.data(n, :) = temp(:)';
                end
                self.N = nBatches;                
                self.K = self.nTags * self.J;
            else
                self.J = 0;
                self.data = given_data;
                self.raw_data = given_data;
            end
            

            % Create storage for other associated matrices
            %self = self.initialize_storage(self.A);
        end
        


        function self = preprocess(self, varargin)
            % Calculates the preprocessing vectors for a block
            %
            % Currently the only preprocessing model supported is to mean center
            % and scale.
            
            % We'd like to preprocess the ``other`` block using settings 
            % from the current block.
            if nargin==2 && self.is_preprocessed
                other = varargin{1};
                if ~isa(other, 'block')
                    error('The new data must be a ``block`` instance.');
                end
                % Don't worry about preprocessing empty blocks.
                if ~isempty(other)
                    mean_center = self.PP.mean_center;
                    scaling = self.PP.scaling;
                    if ~other.is_preprocessed                    
                        other.data = other.data - repmat(mean_center, other.N, 1);
                        other.data = other.data .* repmat(scaling, other.N, 1);
                    end
                else
                    % Catches the case when empty blocks are preprocessed
                    scaling = NaN;
                end
                other.is_preprocessed = true;
                
                % Will scaling introduce missing values?
                if any(isnan(scaling))
                    other.has_missing = true;
                end
                self = other;
                return
            end
            
            % Apply the preprocesing to the training data
            if not(self.is_preprocessed)
                
            
                % Centering based on the mean
                mean_center = nanmean(self.data, 1);
                scaling = nanstd(self.data, 1);

                % Replace zero entries with NaN: this is handled later on with scaling
                % This will create missing data, so we need to set the flag
                % correctly
                if any(scaling < sqrt(eps))
                    scaling(scaling < sqrt(eps)) = NaN;
                    self.has_missing = true;
                end                
                scaling = 1./scaling;
            
            
                self.data = self.data - repmat(mean_center, self.N, 1);
                self.data = self.data .* repmat(scaling, self.N, 1);

                % Store the preprocessing vectors for later on
                self.PP.mean_center = mean_center;
                self.PP.scaling = scaling;  

                self.is_preprocessed = true;
            end
        end
        
        function self = un_preprocess(self, varargin)
            % UNdoes preprocessing for a block.
            %
            % Currently the only preprocessing model supported is to mean center
            % and scale.
            
            % We'd like to preprocess the ``other`` block using settings 
            % from the current block.
            if nargin==2 && self.is_preprocessed
                other = varargin{1};
                if ~isa(other, 'block')
                    error('The new data must be a ``block`` instance.');
                end
                % Don't worry about preprocessing empty blocks.
                if ~isempty(other)
                    mean_center = self.PP.mean_center;
                    scaling = self.PP.scaling;
                    if ~other.is_preprocessed                    
                        other.data = other.data - repmat(mean_center, other.N, 1);
                        other.data = other.data .* repmat(scaling, other.N, 1);
                    end
                end                
                other.is_preprocessed = true;
                
                % Will scaling introduce missing values?
                if any(isnan(scaling))
                    other.has_missing = true;
                end
                self = other;
                return
            end
            
            % Apply the preprocesing to the training data
            if not(self.is_preprocessed)
                
            
                % Centering based on the mean
                mean_center = nanmean(self.data, 1);
                scaling = nanstd(self.data, 1);

                % Replace zero entries with NaN: this is handled later on with scaling
                % This will create missing data, so we need to set the flag
                % correctly
                if any(scaling < sqrt(eps))
                    scaling(scaling < sqrt(eps)) = NaN;
                    self.has_missing = true;
                end                
                scaling = 1./scaling;
            
            
                self.data = self.data - repmat(mean_center, self.N, 1);
                self.data = self.data .* repmat(scaling, self.N, 1);

                % Store the preprocessing vectors for later on
                self.PP.mean_center = mean_center;
                self.PP.scaling = scaling;  

                self.is_preprocessed = true;
            end
        end
        
        function [self, other] = exclude(self, dim, which)
            % Excludes rows (``dim``=1) or columns (``dim``=2) from the block 
            % given by entries in the vector ``which``.
            %
            % The excluded entries are returned as a new block in ``other``.
            % Note that ``self`` is actually returned as a new block also,
            % to accomodate the fact that modelling elements might have been 
            % removed.  E.g. if user excluded rows, then the scores are not
            % valid anymore.
            %
            % Example: [batch_X, test_X] = batch_X.exclude(1, 41); % removes batch 41
            %
            % NOTE: at this time, you cannot exclude a variable from a batch
            % block.  To do that, exclude the variable in the raw data, before
            % creating the block.
            
            
            s_ordinary = struct; 
            s_ordinary.type = '()';
            s_ordinary_remain = struct; 
            s_ordinary_remain.type = '()';
            self_tagnames = self.tagnames;
            if dim == 1 
                if any(which>self.N)
                    error('block:exclude', 'Entries to exclude exceed the size (row size) of the block.')
                end
                s_ordinary.subs = {which, ':'};
                remain = 1:self.N;
                remain(which) = [];
                s_ordinary_remain.subs = {remain, ':'};
                other_tagnames = self.tagnames;
            end
            if dim == 2 
                if any(which>self.K)
                    error('block:exclude', 'Entries to exclude exceed the size (columns) of the block.')
                end
                if strcmp(self.block_type, 'batch')
                    error('block:exclude', 'Excluding tags from batch blocks is not currently supported.')
                end
                s_ordinary.subs = {':', which};
                
                other_tagnames = self.tagnames(which);
                self_tagnames(which) = [];
                remain = 1:self.K;
                remain(which) = [];
                s_ordinary_remain.subs = {':', remain};
            end
            
            if strcmp(self.block_type, 'batch')
                other = block(subsref(self.data, s_ordinary), self.name);
                other.block_type = 'batch';
                other.nTags = self.nTags;
                other.J = self.J;
                other.tagnames = self.tagnames;
                other.raw_data = cell(numel(which), 1);
                for n = 1:numel(which)
                    other.raw_data{n} = self.raw_data{which(n)};
                end
                
                self_raw_data = self.raw_data;
                self_data = subsref(self.data, s_ordinary_remain);
                self = block(self_data, self.name);
                self.block_type = 'batch';
                self.J = other.J;
                self.nTags = other.nTags;
                self.tagnames = other.tagnames;
                self.raw_data = cell(numel(remain), 1);
                for n = 1:numel(remain)
                    self.raw_data{n} = self_raw_data{remain(n)};
                end
            end
        end
        
        function disp(self)
            % Displays a text summary of the block
            if strcmp(self.block_type, 'ordinary')
                fprintf('%s: %d observations and %d variables\n', self.name, self.N, self.K)
            elseif strcmp(self.block_type, 'batch')
                fprintf('%s: %d batches, %d variables, %d time samples (batch unfolded)\n', self.name, self.N, self.nTags, self.J)
            end
                
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
            % plot(block, 'all')             : uses defaults for all plots.
            %                                : Defaults are as shown below
            % plot(block, 'raw', 2, 4)       : raw data in a 2 row, 4 column layout
            % plot(block, 'loadings', 1)     : p_1 as a bar plot
            % plot(block, 'loadings', [1,2]) : p_1 vs p_2 as a scatter plot
            % plot(block, 'weights', ...)    : same as `loadings`, just for weights
            % plot(block, 'onebatch', xx)    : plots scaled values of batch xx

            if nargin == 1
                plottypes = 'raw';
            else
                plottypes = varargin{1};
            end
            
            if strcmpi(plottypes, 'all')
                plottypes = {'raw', 'loadings', 'weights'};
            end
            if ~isa(plottypes, 'cell')
                plottypes = cellstr(plottypes);
            end
            
            if strcmpi(plottypes{1}, 'raw') || strcmpi(plottypes{1}, 'highlight')
                try
                    nrow = floor(varargin{2});
                catch ME
                    nrow = 2;
                end
                try
                    ncol = floor(varargin{3});
                catch ME
                    ncol = 4;
                end
            end
            if strcmpi(plottypes{1}, 'loadings') || strcmpi(plottypes{1}, 'weights') 
                try
                    which_loadings = floor(varargin{2});
                catch ME
                    if self.A > 1
                        which_loadings = [1, 2];
                    elseif self.A == 1
                        which_loadings = 1;
                    elseif self.A <= 0;
                        which_loadings = [];
                    end
                end                
            end
            if strcmpi(plottypes{1}, 'highlight')
                try
                    which_batch = floor(varargin{4});
                catch ME                    
                    which_batch = 1;
                end                
            end
            if strcmpi(plottypes{1}, 'onebatch')
                try
                    which_batch = floor(varargin{2});
                catch ME                    
                    which_batch = 1;
                end    
                try
                    which_tags = floor(varargin{3});
                catch ME                    
                    which_tags = 1:self.nTags;
                end  
            end
                           
            % Iterate over all plots requested by the user
            for i = 1:numel(plottypes)
                plottype = plottypes{i};
                if strcmpi(plottype, 'raw')
                    plot_raw(self, nrow, ncol)
                elseif strcmpi(plottype, 'loadings')
                    plot_loadings(self, which_loadings)                    
                elseif strcmpi(plottype, 'weights')
                    plot_weights(self, which_loadings)
                elseif strcmpi(plottype, 'highlight')
                    plot_highlight_batch(self, nrow, ncol, which_batch)
                elseif strcmpi(plottype, 'onebatch')
                    plot_one_batch(self, which_batch, which_tags)
                end
            end

        end
    end % end methods
end % end classdef
            
%-------- Helper functions. May NOT modify ``self``.
function plot_raw(self, nrow, ncol)
    hA = zeros(nrow*ncol, 1);
    count = -nrow*ncol;
    
    % Ordinary data blocks
    if strcmpi(self.block_type, 'ordinary')
    end
    
   
    if strcmpi(self.block_type, 'batch')
        for j = 1:self.nTags
            if mod(j-1, nrow*ncol)==0
                figure('Color', 'White');
                count = count + nrow*ncol;
            end
            hA(j) = subplot(nrow, ncol, j-count);
        end

        for k = 1:self.nTags
            axes(hA(k))
            for n = 1:self.N
                plot(self.raw_data{n}(:,k),'k'),hold on
            end
            set(hA(k),'FontSize',14)
            axis tight
            grid('on')
            a=axis;
            r = a(4)-a(3);
            axis([1 self.J+1 a(3)-0.1*r a(4)+0.1*r]);
            title(self.tagnames{k})
        end
    end
end

function plot_loadings(self, which_loadings)  
        
    if strcmpi(self.block_type, 'batch')

        nSamples = self.J;                                                              % Number of samples per tag
        nTags = self.nTags;                                                                  % Number of tags in the batch data
        tagNames = char(self.tagnames);
        
        for a = which_loadings
            data = self.P(:, a);
            y_axis_label = ['Loadings, p_', num2str(a)];
            
            data = reshape(data, self.nTags, self.J)';
            cum_area = sum(abs(data));
            data = data(:);
            hF = figure('Color', 'White');
            hA = axes;
            bar(data);

            x_r = xlim;
            y_r = ylim;
            xlim([x_r(1,1) nSamples*self.nTags]);
            tick = zeros(self.nTags,1);
            for k=1:self.nTags
                tick(k) = nSamples*k;
            end

            for k=1:self.nTags
                text(round((k-1)*nSamples+round(nSamples/2)), ...
                     diff(y_r)*0.9 + y_r(1),deblank(tagNames(k,:)), ...
                     'FontWeight','bold','HorizontalAlignment','center');
                text(round((k-1)*nSamples+round(nSamples/2)), ...
                     diff(y_r)*0.05 + y_r(1), sprintf('%.2f',cum_area(k)), ...
                     'FontWeight','bold','HorizontalAlignment','center');
            end

            set(hA,'XTick',tick);
            set(hA,'XTickLabel',[]);
            set(hA,'Xgrid','On');
            xlabel('Batch time repeated for each variable');
            ylabel(y_axis_label);
            pos0 = get(0,'ScreenSize');
            delta = pos0(3)/100*2;
            posF = get(hF,'Position');
            set(hF,'Position',[delta posF(2) pos0(3)-delta*2 posF(4)]);
        end
        
        
    end
end

function plot_weights(self)
end

function plot_highlight_batch(self, nrow, ncol, which_batch)
    hA = zeros(nrow*ncol, 1);
    
    
    % Ordinary data blocks
    if strcmpi(self.block_type, 'ordinary')
        return
    end
    
    count = -nrow*ncol;
    if strcmpi(self.block_type, 'batch')
        for j = 1:self.nTags
            if mod(j-1, nrow*ncol)==0
                figure('Color', 'White');
                count = count + nrow*ncol;
            end
            hA(j) = subplot(nrow, ncol, j-count);
        end

        for k = 1:self.nTags
            axes(hA(k))
            for n = 1:self.N
                plot(self.raw_data{n}(:,k), 'Color', [0.2, 0.2 0.2]),hold on
            end
            plot(self.raw_data{which_batch}(:,k), 'r', 'Linewidth', 1.5)
            set(hA(k),'FontSize',14)
            axis tight
            grid('on')
            a=axis;
            r = a(4)-a(3);
            axis([1 self.J+1 a(3)-0.1*r a(4)+0.1*r]);
            title(self.tagnames{k})
        end
    end
end

function plot_one_batch(self,  which_batch, which_tags)
    
    
    % Ordinary data blocks
    if strcmpi(self.block_type, 'ordinary')
        return
    end
    
    if strcmpi(self.block_type, 'batch')
        figure('Color', 'White');
        hA = axes;
        maxrow = zeros(1, self.nTags)*-Inf;
        minrow = zeros(1, self.nTags)*Inf;
        for n = 1:self.N            
            maxrow = max(maxrow, max(self.raw_data{n}));
            minrow = min(minrow, min(self.raw_data{n}));
        end
            
        
        for k = which_tags
            plot((self.raw_data{which_batch}(:,k)-minrow(k))/maxrow(k),'k', ...
                  'LineWidth',2)
            hold on
            set(hA,'FontSize',14)
            axis tight
            a=axis;
            r = a(4)-a(3);
            axis([1 self.J+1 a(3)-0.1*r a(4)+0.1*r]);
        end
    end
end
