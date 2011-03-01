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
        data_raw = false;       % Raw data for the block: never touched, unless excluded
        data = false;           % Working data: will be processed by the software
        data_pred = [];         % Predicted values of the data: only used in PLS models
        data_pred_pp = [];      % Predicted values of the data, pre-processed: only used in PLS models
        
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
        J = 0;              % Number of batch time steps 
        nTags = 0;          % Number of batch tags
        
        % Block parameters
        A = 0;       % Number of components
        N = 0;       % Number of observations
        K = 0;       % Number of variables (columns)
        
        % Block model parameters
        P = [];      % Block loadings, P.  For PLS this is also C
        T = [];      % Block scores, T.  For PLS this is also U
        T_j = [];    % Instantaneous scores (batch models)
        error_j = [];% Instantaneous errors (batch models)
        S = [];      % Std deviation of the scores
        W = [];      % Block weights (used for PLS only)
        R = [];      % Block weights (W-star matrix, for PLS only) 
        C = [];      % Block loadings (for PLS Y-blocks only)
        U = [];      % Block scores (for PLS Y-blocks only) 
        beta = [];   % Beta-regression coeffients (for PLS X-blocks only)
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
            
            % Prediction of the data, using A components
            % Not available on X-space blocks
            self.data_pred = false;
            
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
                
                self.data_raw = cell(nBatches,1);
                self.data = zeros(nBatches, self.J * self.nTags);
                for n = 1:nBatches
                    startRow = (n-1) * self.J+1;
                    endRow = startRow - 1 + self.J;
                    self.data_raw{n} = given_data(startRow:endRow,:);
                    temp = self.data_raw{n}';
                    self.data(n, :) = temp(:)';
                end
                self.N = nBatches;                
                self.K = self.nTags * self.J;
            else
                self.J = 0;
                self.data = given_data;
                self.data_raw = given_data;
            end
            

            self.A = 0;
        
            % Create storage for other associated matrices
            self = self.initialize_storage(self.A);
        end
        
        function self = initialize_storage(self, A)
            % Initializes storage matrices with the correct dimensions.
            % If ``A`` is given, and ``A`` > ``self.A`` then it will expand
            % the storage matrices to accommodate the larger number of
            % components.
            if A <= self.A
                A = self.A;
            end
            
            self.P = zeroexp([self.K, A], self.P);  % loadings; .C for PLS
            self.T = zeroexp([self.N, A], self.T);  % block scores, .U for PLS
            self.S = zeroexp([1, A], self.S);       % score scaling factors
            self.W = zeroexp([self.K, A], self.W);  % PLS weights
            self.R = zeroexp([self.K, A], self.R);  % PLS weights
            self.C = zeroexp([self.K, A], self.C);  % PLS Y-space loadings
            self.U = zeroexp([self.N, A], self.U);  % PLS Y-space scores

            % Block preprocessing options: resets them
            if numel(self.PP) == 0
                self.PP = struct('mean_center', [], 'scaling', []);
                %self.PP.mean_center = zeros(1, self.K);
                %self.PP.scaling = zeros(1, self.K);
            end

            % Calculated values for this block
            % --------------------------------
            % Overall SPE value for the complete observation, per component
            if isempty(fieldnames(self.stats))
                self.stats.SPE = [];
                self.stats.SPE_j = [];
                
                self.stats.start_SS_col = [];
                self.stats.deflated_SS_col = [];                
                self.stats.R2k_a = [];
                self.stats.R2k_cum = [];
                self.stats.R2_a = [];
                self.stats.R2 = [];
                self.stats.VIP_a = [];
                self.stats.VIP = [];
                
                self.stats.T2 = [];
                self.stats.T2_j = [];
                
                self.stats.model_power = [];
            end
            if isempty(fieldnames(self.lim))
                % Ordinary model portion
                self.lim.t = [];
                self.lim.T2 = [];
                self.lim.SPE = [];
                
                % Instantaneous (batch) model portion
                self.lim.t_j = [];                
                self.lim.SPE_j = []; 
                self.lim.T2_j = []; %not used: we monitoring based on final T2 value
            end

            self.stats.SPE = zeroexp([self.N, A], self.stats.SPE);
            % Instantaneous SPE limit using all A components (batch models)
            self.stats.SPE_j = zeroexp([self.N, self.J], self.stats.SPE_j, true);
                        

            % R^2 per variable, per component; cumulative R2 per variable
            % Baseline value for all R2 calculations: before any components are
            % extracted, but after the data have been preprocessed.
            self.stats.start_SS_col = zeroexp([1, self.K], self.stats.start_SS_col);
            % Used in cross-validation calculations: ssq of each column,
            % per component, after deflation with the a-th component.
            self.stats.deflated_SS_col = zeroexp([self.K, A], self.stats.deflated_SS_col);
            self.stats.R2k_a = zeroexp([self.K, A], self.stats.R2k_a);
            self.stats.R2k_cum = zeroexp([self.K, A], self.stats.R2k_cum);
            % R^2 per block, per component; cumulate R2 for the block
            self.stats.R2_a = zeroexp([A, 1], self.stats.R2_a);
            self.stats.R2 = zeroexp([A, 1], self.stats.R2);

            % VIP value (only calculated for X-blocks); only last column is useful
            self.stats.VIP_a = zeroexp([self.K, A], self.stats.VIP_a);
            self.stats.VIP = zeroexp([self.K, 1], self.stats.VIP);

            % Overall T2 value for each observation
            self.stats.T2 = zeroexp([self.N, 1], self.stats.T2);
            % Instantaneous T2 limit using all A components (batch models)
            self.stats.T2_j = zeroexp([self.N, self.J], self.stats.T2_j);

            % Modelling power = 1 - (RSD_k)/(RSD_0k)
            % RSD_k = residual standard deviation of variable k after A PC's
            % RSD_0k = same, but before any latent variables are extracted
            % RSD_0k = 1.0 if the data have been autoscaled.
            self.stats.model_power = zeroexp([1, self.K], self.stats.model_power);

            % Actual limits for the block: to be calculated later on
            % ---------------------------
            % Limits for the (possibly time-varying) scores
            %siglevels = {'95.0', '99.0'};
            self.lim.t = zeroexp([1, A], self.lim.t);
            self.lim.t_j = zeroexp([self.J, A], self.lim.t_j, true); 

            % Hotelling's T2 limits using A components (column)
            % (this is actually the instantaneous T2 limit,
            % but we don't call it that, because at time=J the T2 limit is the
            % same as the overall T2 limit - not so for SPE!).
            self.lim.T2 = zeroexp([1, A], self.lim.T2);            
            
            % SPE limits for the block and instaneous (i.e. time-varying) limits
            % Overall SPE limit using for ``a`` components (column)
            self.lim.SPE = zeroexp([1, A], self.lim.SPE);
            
            % SPE instantaneous limits using all A components
            self.lim.SPE_j = zeroexp([self.J, 1], self.lim.SPE_j);
            
        end % ``initialize_storage``

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
                other.data_raw = cell(numel(which), 1);
                for n = 1:numel(which)
                    other.data_raw{n} = self.data_raw{which(n)};
                end
                
                self_data_raw = self.data_raw;
                self_data = subsref(self.data, s_ordinary_remain);
                self = block(self_data, self.name);
                self.block_type = 'batch';
                self.J = other.J;
                self.nTags = other.nTags;
                self.tagnames = other.tagnames;
                self.data_raw = cell(numel(remain), 1);
                for n = 1:numel(remain)
                    self.data_raw{n} = self_data_raw{remain(n)};
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
                plot(self.data_raw{n}(:,k),'k'),hold on
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
                plot(self.data_raw{n}(:,k), 'Color', [0.2, 0.2 0.2]),hold on
            end
            plot(self.data_raw{which_batch}(:,k), 'r', 'Linewidth', 1.5)
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
            maxrow = max(maxrow, max(self.data_raw{n}));
            minrow = min(minrow, min(self.data_raw{n}));
        end
            
        
        for k = which_tags
            plot((self.data_raw{which_batch}(:,k)-minrow(k))/maxrow(k),'k', ...
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
