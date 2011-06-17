% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% Licensed under the BSD license.
% -------------------------------------------------------------------------
%
% Code for plotting and interrogating latent variable models.


% Create a handle class
% This is the highest level latent variable model class

classdef lvmplot < handle
    properties
        hF = -1;            % Figure window
        hA = [];            % Vector of axes [nRow x nCol] (in subplot order)
        hS = [];            % Vector of plot handles [nRow x nCol] (in subplot order) for data being shown
        hM = [];            % Vector of axis selector markers
        hDropdown = [];     % Java dropdown
        hMarker = [];       % Handle to the marker
        nA = NaN;           % Number of axes in plot
        nRow = NaN;         % Number of rows of axes
        nCol = NaN;         % Number of columns of axes
        index = -1;         % Which entry in self.hA is the current axis?
        
        model = [];         % The model being plotted
        dim = NaN;          % Which dimension of the data is being plotted
        screen = get(0,'ScreenSize');               
        
        ptype = '';         % Plot type (base type)
        
        gca = -1;           % This is the handle to the current axis
        hFoot = -1;         % Handle to the footer        
        hSelectX = -1;      % Handle to the X-axis selection panel
        hSelectY = -1;      % Handle to the Y-axis selection panel
        
        registered = {};    % Cell array of registered plots
                            % FUTURE: make this a struct (easier to code for)
                
        c_block = NaN;      % Current block
        
        opt = struct;       % Default plotting options
    end 
    
    % Only set by this class and subclasses
    properties (SetAccess = protected)
    end
    
    % Subclasses may choose to redefine these methods
    methods
        function self = lvmplot(model, varargin)
            % Class cconstructor
            self.set_defaults();
            self.model = model;
            if nargin > 1
                self.get_registered_plots()
                self.ptype = varargin{1};
            else
                self.start_plotter();
            end
        end
        
        function self = set_defaults(self)
            self.opt.highlight = [255, 102, 0]/255;  % orangey colour
            self.opt.font.size = 14;
            self.opt.font.name = 'Arial';
            self.opt.fig.backgd_colour = [1, 1, 1];
            self.opt.fig.add_menu = true;
            self.opt.fig.default_visibility = 'off';
            self.opt.add_footer = true;
            self.opt.show_labels = true;
            self.opt.bar.facecolor = [0.25, 0.5, 1];
        end
        
        function out = get.gca(self)
            out = self.hA(self.index);
            set(self.hF, 'CurrentAxes', out);
        end
        
        function h = get.hF(self)
            h = self.hF;
        end
        
        function start_plotter(self)
            % Quick plot figure
            
            W = 250;
            H = 300 - 55;
            delta = 5;
            bheight = 30;
            if self.model.M > 0
                H = H + 3 * (bheight+delta);
            end
            hQuick = figure(...
                'ButtondownFcn'    ,'', ...
                'Color'            ,get(0, 'defaultuicontrolbackgroundcolor'), ...  % 'color',get(0,'FactoryUicontrolBackgroundColor')
                'Colormap'         ,[], ...
                'IntegerHandle'    ,'on', ...
                'InvertHardcopy'   ,'off', ...
                'HandleVisibility' ,'on', ...
                'Menubar'          ,'none', ...
                'NumberTitle'      ,'off', ...
                'PaperPositionMode','auto', ...
                'Resize'           ,'off', ...
                'Visible'      ,'on', ...
                'WindowStyle'      ,'normal', ...
                'Name'             ,'Please choose a plot type', ...
                'Position'         ,[0 0 W H], ...
                'Tag'              ,'PLOT_GUI', ...
                'Interruptible'    ,'off');
           
       
            movegui(hQuick,'west');
            
            ctl = struct;
            ctl.Interruptible = 'off';
            ctl.BusyAction = 'queue';
            ctl.Parent = hQuick;
            ctl.Units = 'pixels';
            ctl.FontSize = 12;
            
            % Observation-based plots
            % -----------------------
            
            nbuttons = 2;
            if self.model.M > 0
                nbuttons = 3;
            end
            offset = 25;            
            h_obs = nbuttons*(bheight + delta) + offset;
            w = W-10;            
            pObs = uipanel(ctl, 'Title', 'Observation-based plots', ...
                'Position', [delta H-h_obs w h_obs]);
            
            ctl.Parent = pObs;
            idx = 1;
            uicontrol(ctl, ...
                'String', 'Scores',...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_obs-offset-idx*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, 'scores'));
            idx = idx + 1;
            uicontrol(ctl, ...
                'String', 'SPE',...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_obs-offset-idx*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, 'SPE'));
            idx = idx + 1;
            
            if self.model.M > 0
                uicontrol(ctl, ...
                    'String', 'Predictions',...
                    'Style', 'Pushbutton', ...
                    'Position', [delta, h_obs-offset-idx*(bheight+delta)+delta, w-2*delta, bheight], ...
                    'Callback', @(src,event)plot(self.model, 'predictions'));
            end
            
            % Variable-based plots
            % -----------------------
            if self.model.M > 0
                plot_str = 'Weights';
                nbuttons = 5;
            else
                plot_str = 'Loadings';
                nbuttons = 3;
            end
            
            h_var = nbuttons*(bheight + delta) + offset;
            ctl.Parent = hQuick;
            pVar = uipanel(ctl, 'Title', 'Variable-based plots', ...
                'Position', [delta H-h_obs-delta-h_var w h_var]);
            
            ctl.Parent = pVar;
            idx = 1;
            uicontrol(ctl, ...
                'String', plot_str, ...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_var-offset-idx*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, plot_str));
            idx = idx + 1;
            
            uicontrol(ctl, ...
                'String', 'VIP plot',...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_var-offset-idx*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, 'VIP'));
            idx = idx + 1;
            if self.model.M > 0
                uicontrol(ctl, ...
                    'String', 'Coefficient plots',...
                    'Style', 'Pushbutton', ...
                    'Position', [delta, h_var-offset-idx*(bheight+delta)+delta, w-2*delta, bheight], ...
                    'Callback', @(src,event)plot(self.model, 'Coefficient'));
                idx = idx + 1;
                uicontrol(ctl, ...
                    'String', 'R2 (per Y-variable)',...
                    'Style', 'Pushbutton', ...
                    'Position', [delta, h_var-offset-idx*(bheight+delta)+delta, w-2*delta, bheight], ...
                    'Callback', @(src,event)plot(self.model, 'R2-Y-variable'));
                idx = idx + 1;
            end
            
            uicontrol(ctl, ...
                'String', 'R2 (per variable)',...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_var-offset-idx*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, 'R2-variable'));
            
            % Component-based plots
            % -----------------------
%             nbuttons = 1;
%             h_cmp = nbuttons*(bheight + delta) + offset;
%             ctl.Parent = hQuick;
%             pCmp = uipanel(ctl, 'Title', 'Component-based plots', ...
%                 'Position', [delta H-h_obs-delta-h_var-delta-h_cmp w h_cmp]);
%             
%             ctl.Parent = pCmp;
%             uicontrol(ctl, ...
%                 'String', 'R2 (per component)', ...
%                 'Style', 'Pushbutton', ...
%                 'Position', [delta, h_cmp-offset-1*(bheight+delta)+delta, w-2*delta, bheight], ...
%                 'Callback', @(src,event)plot(self.model, 'R2-component'));
            
           
            
            set(hQuick, 'Visible' ,'on')
        end
        
        function clear_figure(self)
            N = numel(self.hA);
            for n = 1:N
                if ishandle(self.hA(n))
                    delete(self.hA(n))
                end
                if ishandle(self.hS(n))
                    delete(self.hS(n))
                end
                if ishandle(self.hM(n))
                    delete(self.hM(n))
                end
            end
        end
        
        function h = new_figure(self, varargin)
            % Syntax: self.new_figure('off')
            % Creates a new figure; set the visibility for it
            % (Visibility off is useful for unit testing)
            
            self.hF = figure(...
                 'Visible',     'on', ...
                 'DoubleBuffer',  'on', ...  % Allows fast redrawing of images
                 'BackingStore', 'off');     % Faster animation when figures are updated frequently
                 %'WindowButtonDownFcn', @mouseclick_callback);
                 

            set(self.hF, 'Color', self.opt.fig.backgd_colour);
            if self.opt.fig.add_menu
                set(self.hF, 'ToolBar', 'figure');
            end
            units = get(self.hF, 'Units');
            set(self.hF, 'units', 'Pixels');
             
            fPos = get(self.hF, 'position');
            fPos(1) = round(0.05*self.screen(3));
            fPos(2) = round(0.1*self.screen(4));
            fPos(3) = 0.70*self.screen(3);
            fPos(4) = 0.80*self.screen(4);
            set(self.hF, 'Position', fPos);
  
            set(self.hF, 'Units', 'normalized')
            Txt.Interruptible = 'off';  
            Txt.BusyAction    = 'queue';
            Txt.Style         = 'text';
            Txt.Units         = 'normalized';
            Txt.HorizontalAli = 'center';
            Txt.Background    = self.opt.fig.backgd_colour;
            Txt.Parent        = self.hF;
            Txt.FontSize = 10;
            
            % Footer
            % -------
            self.hFoot = uicontrol(Txt, ...
                             'Position',[0.02, 0.00, 0.96 0.03], ...
                             'ForegroundColor',[0 0 0], 'String', '', ...
                             'HorizontalAlignment', 'left');
            set(self.hFoot, 'Units', 'pixels');
            
            % Block dropdown
                         
            % Add a contribution toolbar icon
            icon = fullfile(matlabroot,'/toolbox/matlab/icons/greenarrowicon.gif');
            [cdata,map] = imread(icon);
 
            % Convert white pixels into a transparent background
            map(find(map(:,1)+map(:,2)+map(:,3)==3)) = NaN;
 
            % Convert into 3D RGB-space
            cdataRedo = ind2rgb(cdata,map);
            % Add the icon to the latest toolbar
            hToolbar = findall(self.hF, 'tag','FigureToolBar');
            hSimulate = uipushtool('Parent', hToolbar, 'cdata',cdataRedo, ...
                'tooltip','Contribution tool', 'ClickedCallback', @(src,event)get_contribution(self.model, self));
            set(hSimulate,'Separator','on');

            
            % Block selector panel
            % ------------
            % From: http://undocumentedmatlab.com/blog/figure-toolbar-components/
            % and : http://www.mathworks.com/matlabcentral/newsreader/view_thread/294487
            jToolbar = get(get(hToolbar,'JavaContainer'),'ComponentPeer');
            if ~isempty(jToolbar)
                block_names = cell(self.model.B+1, 1);            
                for b = 1:self.model.B
                    block_names{b+1} = self.model.blocks{b}.name;
                end
                block_names{1} = 'Overall';
                dropdown = javax.swing.JComboBox(block_names);
                self.hDropdown = handle(dropdown, 'CallbackProperties');
                set(self.hDropdown, 'ActionPerformedCallback', @dropdown_block_selector);
                jToolbar(1).add(self.hDropdown,0); %5th position, after printer icon
                jToolbar(1).repaint;
                jToolbar(1).revalidate;
                self.hDropdown.name = num2str(self.hF); % so we can get access to ``self`` later
                
            end
            self.c_block = 0;  % The default block is the superblock
            
            set(self.hF, 'units', units);
            if nargin > 1
                set(self.hF, 'Visible', varargin{1});
            else
                set(self.hF, 'Visible', self.opt.fig.default_visibility)
            end
            h = self.hF;
            
            % Store a reference to ``self`` in the figure's UserData
            set(self.hF, 'UserData', self)
        end
        
        function h = new_axis(self, n, varargin)
            % Syntax: self.new_axis(n)
            % Creates a new axis in position ``n`` in the figure window
            % given by handle ``self.hF``
            % This position ``n`` is the same as the subplot(row, col, n).
            
            for i = n
            
                h = subplot(self.nRow, self.nCol, i);            
                %h = axes('Parent', self.hF, );  % NextPlot = replace
                set(h, 'box', 'on')
                set(h, 'Color', self.opt.fig.backgd_colour)
                set(h, 'FontName', self.opt.font.name)
                set(h, 'FontSize', self.opt.font.size)
                set(h, 'ButtonDownFcn', @self.mouseclick_callback);
                self.hA(i) = h;
                self.index = i;

                % Add axis selector at the bottom left corner
                % Set "hover" tooltip also
                units = get(h, 'Units');
                set(h, 'Units', 'pixels');
                hPos_pix = get(h, 'Position');
                set(h, 'Units', units);
                ctl.Interruptible = 'off';  
                ctl.BusyAction    = 'queue';
                ctl.Style         = 'text';
                ctl.Units         = 'pixels';
                ctl.HorizontalAli = 'center';
                ctl.Style         = 'togglebutton';
                ctl.Background    = self.opt.fig.backgd_colour;
                ctl.Parent        = self.hF;
                ctl.FontSize = 6;
                self.hM(i) = uicontrol(ctl, ...
                    'String', 'AA', ...
                    'ForegroundColor',[0 0 0], ...
                    'Position',[hPos_pix(1)-50, hPos_pix(2)-25, 15 15], ...
                    'TooltipString', 'Adjust axes', ...
                    'Callback', @adjust_axes);
                set(self.hM(i), 'Units', 'Normalized')
                
                
                % How many plot elements will there be?
                if self.dim == 0
                    num = self.model.A;
                else
                    if self.c_block > 0
                        num = self.model.blocks{self.c_block}.shape(self.dim);
                    else
                        % Superblock plots:
                        if self.dim == 1
                            num = size(self.model.super.T, 1);
                        elseif self.dim == 2;
                            num = size(self.model.super.T, 2);
                        end
                    end
                end
                vector = zeros(num, 1) .* NaN;
                set(h, 'Nextplot', 'add')
                self.hS(self.index) = plot(h, vector, vector);
                set(self.hS, 'tag', 'lvmplot_series')

                series.x_type = {};        % Registration entry: what is on the x-axis; lookup in self.registered
                series.y_type = {};        % Registration entry: what is on the y-axis; lookup in self.registered
                series.x_num  = -1;        % Which entry from the dropdown in shown on the x-axis
                series.y_num  = -1;        % Which entry from the dropdown in shown on the y-axis
                setappdata(h, 'SeriesData', series)
            end

        end
        
        function set_plot(self, idx, xaxis, yaxis)
            % Sets what is plotted on each axis for the current axis handle,
            % found by the ``idx`` into self.hA. The ``dim`` tells us which
            % dimension is being plotted.
            
            h = self.hA(idx);
            series = getappdata(h, 'SeriesData');
            [series.x_type, series.x_num] = self.validate_plot(xaxis);
            [series.y_type, series.y_num] = self.validate_plot(yaxis);
            setappdata(h, 'SeriesData', series);
        end
        
        function [plottype, entry_num] = validate_plot(self, request)
            % Validates the ``request``ed plot and sees if it exists in the
            % registered plot types for that ``dim``ension.
            subset_idx = cell2mat(self.registered(:,3)) == self.dim;
            subset = self.registered(subset_idx, :);
            
            for i = 1:size(subset, 1)
                if strcmpi(subset{i,1}, request{1})
                    plottype = subset(i,:);
                    entry_num = validate_plot_index(self, subset(i,:), request{2});
                    return
                end
            end
            error('lvmplot:validate_plot', 'Requested plot not found')
        end
        
        function entry = validate_plot_index(self, plot_entry, index)
            % Returns all valid indicies for ``plot_entry`` using the
            % ``plt.more_type`` registration
            entry = NaN;
            
            switch plot_entry{6}
                case {'a', '1:A'}
                    A = self.model.A;
                    if index <= A
                        entry = index;
                    end
                case '<model>';
                    if index == -1;
                        entry = index;
                    end
                    
                case '<order>'
                    if index == -1;
                        entry = index;
                    end
                    
                case '<label>';
                    % See which dimension we are plotting in and see if we can
                    % index into that dimension
                    
                    % TODO(KGD): For now we can only show variables from the 
                    % Y-block
                    if strcmp(plot_entry(:,7), 'Y')
                        shape = self.model.Y.shape(2);
                        if index <= shape
                            entry = index;
                        end
                    end
                    
            end
            
            if isnan(entry)
                error('lvmplot:validate_plot_index', 'Invalid index supplied')
            end
        end
        
        function add_text_labels(hA, x, y, labels)
            % Adds text labels to axis ``hA`` at the ``x`` and ``y`` coordinates
            if popt.show_labels
                extent = axis(hA);
                delta_x = 0.01*diff(extent(1:2));
                delta_y = 0.01*diff(extent(3:4));
                for n = 1:numel(x)
                    text(x+delta_x, y+delta_y, labels(n), 'parent', hA)
                end
            end
        end % ``add_text_labels``
        
        function get_registered_plots(self)
            plts = self.model.register_plots();
            self.registered = {};
            for j = 1:numel(plts)
                self.registered{end+1, 1} = plts(j).name;
                self.registered{end, 2} = plts(j).weight;
                self.registered{end, 3} = plts(j).dim;
                self.registered{end, 4} = plts(j).callback;
                self.registered{end, 5} = plts(j).more_text;
                self.registered{end, 6} = plts(j).more_type;
                self.registered{end, 7} = plts(j).more_block;
                self.registered{end, 8} = plts(j).annotate;
            end           
            
        end
        
        function update_all_plots(self)
            % Goes through all plots in the figure and calls their callback
            % function to update the plot, using the latest values in the 
            % dropdowns
            % Adds annotations to all plots
            for i = 1:numel(self.hA)
                self.index = i;
                hAx = self.gca;
                
                series = getappdata(hAx, 'SeriesData');
                
                hChild = get(hAx, 'Children');
                for h = 1:numel(hChild)
                    if ishandle(hChild(h))
                        if ~strcmpi(get(hChild(h), 'Tag'), 'lvmplot_series')
                            delete(hChild(h))
                        end
                    end
                end
                
                % Get the callback function
                cb_update_x = series.x_type{4};
                cb_update_y = series.y_type{4};
                
                % Call the plotting callbacks to do the work
                series.current = 'x';
                cb_update_x(self, series);
                series.current = 'y';
                cb_update_y(self, series);
            end
        end
        
        function update_annotations(self)
            % Adds annotations to all plots
            for i = 1:numel(self.hA)
                self.index = i;
                hAx = self.gca;
                series = getappdata(hAx, 'SeriesData');
                cb_annotate_x = series.x_type{8};
                cb_annotate_y = series.y_type{8};
                
                % Call the annotate callback to do the work
                if ~isempty(cb_annotate_x)
                    series.current = 'x';
                    cb_annotate_x(self, series)
                end
                if ~isempty(cb_annotate_y)
                    series.current = 'y';
                    cb_annotate_y(self, series)
                end
                
                extent = axis;
                set(hAx, 'Nextplot', 'add')
                hd = plot([0, 0], [-1E10, 1E10], 'k', 'linewidth', 2);
                set(hd, 'tag', 'vline', 'HandleVisibility', 'on');   
                hd = plot([-1E50, 1E50], [0, 0], 'k', 'linewidth', 2);
                set(hd, 'tag', 'hline', 'HandleVisibility', 'on');

                xlim(extent(1:2))
                ylim(extent(3:4)) 
            end
            
        end
        
        function label_scatterplot(self, hPlot, labels)
            % Add labels to a scatter plot series, given by ``hPlot`` and
            % labels in cell array ``labels``
            if ~self.opt.show_labels || isempty(labels)
                return
            end
            
            %hAx = get(hPlot, 'Parent');
            x_data = get(hPlot, 'XData');
            y_data = get(hPlot, 'YData');
            
            if  numel(x_data) ~= numel(labels)
                % KGD(HACK): to get batch plots working
                %error('lvmplot:label_scatterplot', 'Incorrect number of labels supplied')
            end
            
            axis_extent = axis;
            max_range = axis_extent(4) - axis_extent(3); %max(max(y_data),  - extent(3));
            delta = max_range * 0.02;
            fontweight = 'normal';
            
            for n = 1:numel(x_data)
                 hText = text(x_data(n), y_data(n)+delta, strtrim(labels{n}), 'Rotation', 0);
                 set(hText, 'FontSize', 12, 'FontWeight', fontweight, ...
                     'HorizontalAlignment', 'center')                 
            end
            
        end % ``label_scatterplot``
        
        function extent = get_good_limits(self, data, extent, varargin) %#ok<MANU>
            % Finds good axis limits for the ``data`` for changing the axis
            % limits, ``extent``.
            
            % Make this function smarter with options for
            %    'symmetrical' : i.e. symmetrical about zero (e.g. for loadings plot)
            %    'equal'       (e.g. for loadings plot; for obs-pred plots)
            %    'min_zero' (e.g. for SPE, T2, VIP plot')
            %    'include_zero' (e.g. for score line plots)
            
            data = sort(data(:));
            range = data(end)-data(1);
            actual_range = extent(2) - extent(1);
            if actual_range > 1.4*range
                delta = 0.15*range;
                extent(1) = data(1)-delta;
                extent(2) = data(end)+delta;
            end
            if nargin==4
                options = varargin{1};                
                switch lower(options)
                    case 'zero'
                        extent(1) = min(0.0, extent(1));
                end
            end
            if data(end) > extent(2)
                extent(2) = data(end) + (data(end) - data(1))*0.05;
            end
            if data(1) < extent(1)
                extent(1) = data(1) - (data(end) - data(1))*0.05;
            end
        end % ``get_good_limits``
        
        function hPlot = set_data(self, hAx, x_data, y_data)
            % Sets the x_data and y_data in the current axes, ``hAx``            
            
            hPlot = findobj(hAx, 'Tag', 'lvmplot_series');
            if ~isempty(hPlot)
                if ~isempty(x_data)
                    set(hPlot, 'XData', x_data);
                    set(hAx, 'XLim', self.get_good_limits(x_data, get(hAx, 'XLim')))
                end
                if ~isempty(y_data)
                    set(hPlot, 'YData', y_data);
                    set(hAx, 'YLim', self.get_good_limits(y_data, get(hAx, 'YLim')))
                end
            else
                set(hAx, 'Nextplot', 'add');
                if isempty(y_data)
                    hPlot = plot(hAx, x_data, x_data.*NaN);
                elseif isempty(x_data)
                    hPlot = plot(hAx, y_data.*NaN, y_data);
                else
                    hPlot = plot(hAx, x_data, y_data);
                end
                set(hPlot, 'Tag', 'lvmplot_series')                
            end            
        end
        
        function mouseclick_callback(self, varargin)    
            hAx = varargin{1};
            cPoint = get(hAx, 'CurrentPoint');
            hObj = findobj(hAx, 'Tag', 'lvmplot_series');
            if numel(hObj) == 1 && strcmp(get(hObj,'Type'), 'line')
                x_data = get(hObj, 'XData');
                y_data = get(hObj, 'YData');
                distance = sqrt((x_data-cPoint(1,1)).^2 + (y_data-cPoint(1,2)).^2);
                [value, idx] = min(distance); %#ok<ASGLU>
                
                if isempty(self.hMarker) || not(ishandle(self.hMarker))
                    set(hAx, 'Nextplot', 'add')
                    self.hMarker = plot(hAx, x_data(idx), y_data(idx), 'rh', ...
                        'MarkerSize', 15, 'LineWidth', 1.5, ...
                        'UserData', idx);
                    setappdata(hAx, 'Marker', self.hMarker)
                else
                    set(self.hMarker, 'XData', x_data(idx), 'YData', y_data(idx), ...
                        'UserData', idx, 'Parent', hAx)
                end
                extent = [get(hAx,'XLim') get(hAx,'YLim')];
                if cPoint(1,1)<extent(1) || cPoint(1,1)>extent(2) || cPoint(1,2)<extent(3) || cPoint(1,2)>extent(4)
                    delete(self.hMarker)
                    rmappdata(hAx, 'Marker');
                end
            end
            
        end
        
        function annotate_batch_trajectory_plots(self, hAx, hBar, batch_block)
            % Annotates a batch-like trajectory plot in the axis ``hAx``
            % given the series handle, ``hBar`` and the corresponding batch
            % block from which it came, ``batch_block`` (used for dimensioning)
            
            data = get(hBar, 'YData');
            set(hAx, 'Xlim', self.get_good_limits(get(hBar, 'XData'), get(hAx, 'XLim')))
            set(hAx, 'Ylim', self.get_good_limits(data, get(hAx, 'YLim'), 'zero'))
            nSamples = batch_block.J;     % Number of samples per tag
            nTags = batch_block.nTags;    % Number of tags in the batch data
            
            % Reshape the data in the bar plot to variable based order
            data = reshape(data, nTags, nSamples)';
            data(isnan(data)) = 0.0;
            cum_area = sum(abs(data));
            set(hBar, 'YData', data(:), 'FaceColor', self.opt.bar.facecolor, ...
                                        'EdgeColor', self.opt.bar.facecolor);
                                    
            tagNames = batch_block.labels{2};
            
            x_r = xlim;
            y_r = ylim;
            xlim([x_r(1,1) nSamples*nTags]);
            tick = zeros(nTags,1);
            for k=1:nTags
                tick(k) = nSamples*k;
            end
            set(hAx, 'LineWidth', 1);

            for k=1:nTags
                text(round((k-1)*nSamples+round(nSamples/2)), ...
                    diff(y_r)*0.9 + y_r(1),strtrim(tagNames(k,:)), ...
                    'FontWeight','bold','HorizontalAlignment','center');
                text(round((k-1)*nSamples+round(nSamples/2)), ...
                    diff(y_r)*0.05 + y_r(1), sprintf('%.1f',cum_area(k)), ...
                    'FontWeight','bold','HorizontalAlignment','center');
            end

            set(hAx,'XTick',tick);
            set(hAx,'XTickLabel',[]);
            set(hAx,'Xgrid','On');
            xlabel('Batch time repeated for each variable');            
            
        end % ``annotate_batch_trajectory_plots``
        
    end % end: methods (ordinary)
    
    methods (Static=true)
                
        function [nrow, ncol] = subplot_layout(nplots) 
            % Determines the best layout for ``nplots``.  
            % TODO(KGD): give the figure handle so that it can take a minimum
            % plot size into account
            switch nplots
                case 1
                    layout = [1, 1];
                case 2
                    layout = [1, 2];
                case 3
                    layout = [1, 3];
                case 4
                    layout = [2, 2];
                case {5, 6}
                    layout = [2, 3];
                case {7, 8}
                    layout = [2, 4];
                case {9, 10}
                    layout = [2, 5];
                otherwise
                    layout = [2, 6];
            end
            nrow = layout(1);
            ncol = layout(2);
        end
        
        function annotate_barplot(hBar, labels, varargin)
            % Adds vertical ``labels`` to bar plot series, ``hBar``
            isstacked = false; 
            if nargin==3
                if strcmp( varargin{1}, 'stacked')
                    isstacked=true;
                    hBar = hBar(1);
                end
            end
            hAx = get(hBar, 'Parent');
            set(hAx, 'XTickLabel', {})
            x_data = get(hBar, 'XData');
            y_data = get(hBar, 'YData');
            
            set(hAx, 'XLim', [x_data(1)-1, x_data(end)+1])
            axis_extent = axis;
            max_range = axis_extent(4) - axis_extent(3); %max(max(y_data),  - extent(3));
            delta = max_range * 0.03;
            if numel(x_data) > 50
                fontweight = 'normal';
            else
                fontweight = 'bold';
            end
            if isstacked                
                point = (axis_extent(4) - axis_extent(3))*0.05 + axis_extent(3);
                y_data = ones(size(x_data)) .* point;
            end
            set(hAx, 'XTick', [])
            
            if numel(x_data) ~= numel(labels)
                % TODO(KGD): this occurs with batch blocks
                % Figure out a way to elegantly deal with batck blocks
                return
                %error('lvmplot:annotate_barplot', 'Incorrect number of labels supplied')
            end
            
            for n = 1:numel(x_data)
                hText = text(x_data(n), y_data(n)+delta, labels{n}, 'Rotation', 90);
                set(hText, 'FontSize', 12, 'FontWeight', fontweight)
                t_extent = get(hText, 'Extent');
                if (t_extent(2) + t_extent(4)) > axis_extent(4)
                    set(hText, 'Position', [n, y_data(n)-delta-t_extent(4), 0])
                end
            end            
        end %``annotate_barplot``
      
    end % end: methods (static)
end % end: ``classdef``

function dropdown_block_selector(hCombo, varargin)
    idx = get(hCombo,'SelectedIndex');  % 0=topmost item
    self = get(str2double(hCombo.getName), 'UserData');
    self.c_block = idx;
    self.update_all_plots();
    self.update_annotations();
end

function dropdown_callback_level_1(varargin)
    % A dropdown in the figure has changed; update plot
    hPanel = get(gcbo, 'Parent');
    callbacks = get(hPanel, 'UserData');
    selected = get(gcbo, 'Value');
    
    disp('Callback changed')
end

function adjust_axes(varargin)
    hOb = varargin{1};
    self = get(get(hOb, 'Parent'), 'UserData');
    
    % User wants to change axes
    if get(hOb, 'Value') == 1
        
        this_dim = cell2mat(self.registered(:,3)) == self.dim;
        dropdown_str = self.registered(this_dim,1);
        dropdown_weight = cell2mat(self.registered(this_dim, 2));
        registered_callback = self.registered(this_dim, :);
        [dropdown_weight, idx] = sort(dropdown_weight, 1, 'descend');
        dropdown_strs = dropdown_str(idx);
        registered_callbacks = registered_callback(idx, :);
        
        
        hAx = self.gca;
        set(hAx, 'Unit', 'Normalized')
        hAx_pos = get(hAx, 'Position');
        units = get(self.hF, 'Units');
        set(self.hF, 'Units', 'Normalized')
        
        ctl.Interruptible = 'off';
        ctl.BusyAction = 'queue';
        ctl.Parent = self.hF;
        ctl.Units = 'Normalized';
        ctl.FontSize = 12;
        H = 0.95*hAx_pos(3);
        W = 0.95*hAx_pos(4);
       
        % X-axis panel
        % ------------
        hFr = uipanel(ctl, 'Title', 'X-axis series', ...
            'Position', [H W H/0.95*0.10 H/0.95*0.35], ...
            'BorderType', 'beveledout', ...
            'TitlePosition', 'centertop');
        set(hFr, 'Units', 'pixels')
        hFr_pos = get(hFr, 'Position');
        set(hFr, 'Position', [hFr_pos(1) hFr_pos(3), 200, 50])
        self.hSelectX = hFr;
        set(self.hSelectX, 'UserData', registered_callbacks);
        
        ctl.Parent = self.hSelectX;
        ctl.units = 'Normalized';
        uicontrol(ctl, 'Style', 'popupmenu', ...
            'String', dropdown_strs, ...
            'Callback', @dropdown_callback_level_1, ...
            'Position', [0, 0, 00.70, .9]);
        
        
        set(self.hF, 'Units', units);
        
        
%         
%         % Y-axis panel
%         ctl.Units = 'pixels';
%         hFr = uipanel(ctl, 'Title', 'Y-axis series', ...
%             'Position',[hFr_pos(4) hFr_pos(2), 200, 50], ...
%             'BorderType', 'beveledout', ...
%             'TitlePosition', 'centertop');        
%         self.hSelectY = hFr;
%         
%         ctl.Parent = self.hSelectY;
%         uicontrol(ctl, 'Style', 'popupmenu', ...
%             'String', dropdown_strs, ...
%             'Callback', @dropdown_callback_level_2, ...
%             'Position', [0, 0, 00.70, .9]);
%             
        
        
        
        
    % User wants to finish axis change: i.e. just hide the dropdowns
    elseif get(hOb, 'Value') == 0
        if ishandle(self.hSelectX)
            delete(self.hSelectX);
        end
        if ishandle(self.hSelectY)
            delete(self.hSelectY);
        end
    end
end
