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
        function self = block_batch(given_data, block_name, varargin)
            % Initialize the superclass
            self = self@block_base([], []);
            
            % Batch data blocks are always 3-dimensional
            self.labels = cell(3, 0); 
            
            nBatches = -1;
            
            % Third and subsequent arguments: 
            if nargin > 2
                for i = 1:numel(varargin)
                    key = varargin{i}{1};
                    value = varargin{i}{2};
                    
                    if strcmpi(key, 'nbatches')
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
                    end
                end
            end
            if nBatches < 0
                error('block_batch:number_of_batches_not_specified', ...
                     ['Please specify the number of batches. For example: ', ...
                      ' {"nbatches", 12} '])
            end
            
            self.N = size(given_data, 1);
            self.K = size(given_data, 2);
            
                   
            % Reshape the data
            self.J = self.N / nBatches;
            self.nTags = self.K;
            if self.J ~= floor(self.J)
                error('block_batch:inconsistent_data_specification',['There are %d rows and you ', ...
                    'specified %d batches.  This is not consistent with the data provided.'], ...
                    self.N, nBatches)
            end

            % Unfold the data:
            % Unfolded order is to have all tags at time 1, then all the 
            % tags at time 2, and so on, up till time step J.
            % So we write that X = K*J (K tags, at J time steps).

            self.batch_raw = cell(nBatches,1);
            self.data = zeros(nBatches, self.J * self.nTags);
            for n = 1:nBatches
                startRow = (n-1) * self.J+1;
                endRow = startRow - 1 + self.J;
                self.batch_raw{n} = given_data(startRow:endRow,:);
                temp = self.batch_raw{n}';
                self.data(n, :) = temp(:)';
            end
            self.N = nBatches;                
            self.K = self.nTags * self.J;
            
            % Missing data handling
            self.mmap = false;            
            missing_map = ~isnan(self.data);   % 0=missing, 1=present
            if all(missing_map)
                self.has_missing = false;
            else
                self.has_missing = true;
                self.mmap = missing_map;
            end
            
            % Block name
            if isempty(block_name)
                self.name = ['batch-', num2str(self.N), '-', num2str(self.K), '-', num2str(self.J)];
                self.name_type = 'auto';
            else
                self.name = block_name;
                self.name_type = 'given';
            end
            
        end
        
        function disp_header(self)
            % Displays a text summary of the block
            fprintf('%s: %d batches, %d variables, %d time samples (batch unfolded)\n', self.name, self.N, self.nTags, self.J)
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
