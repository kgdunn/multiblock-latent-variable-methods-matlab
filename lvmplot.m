% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for plotting and interrogating latent variable models.


% Create a handle class
% This is the highest level latent variable model class

classdef lvmplot < handle
    properties
        hF = -1;            % Figure window
        hA = [];            % Vector of axes [nRow x nCol] (in subplot order)
        hM = [];            % Vector of axis selector markers
        nA = NaN;           % Number of axes in plot
        nRow = NaN;         % Number of rows of axes
        nCol = NaN;         % Number of columns of axes
        index = -1;         % Which entry in self.hA is the current axis?
        gca = -1;           % This is the handle to the current axis
        model = [];         % The model being plotted
        dim = NaN;          % Which dimension of the data is being plotted
        screen = get(0,'ScreenSize');               
        
        ptype = '';         % Plot type (base type)
        
        hFoot = -1;         % Handle to the footer
        
        hSelectX = -1;
        hSelectY = -1;
        
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
                self.ptype = varargin{1};
            else
                self.start_plotter();
            end
        end
        
        function self = set_defaults(self)
            self.opt.highlight = [255, 102, 0]/255;  % orangy colour
            self.opt.font.size = 14;
            self.opt.font.name = 'Arial';
            self.opt.fig.backgd_colour = [1, 1, 1];
            self.opt.fig.add_menu = true;
            self.opt.fig.default_visibility = 'off';
            self.opt.add_footer = true;
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
            H = 300;
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
            delta = 5;
            bheight = 30;
            nbuttons = 3;
            offset = 25;            
            h_obs = nbuttons*(bheight + delta) + offset;
            w = W-10;            
            pObs = uipanel(ctl, 'Title', 'Observation-based plots', ...
                'Position', [delta H-h_obs w h_obs]);
            
            ctl.Parent = pObs;
            uicontrol(ctl, ...
                'String', 'Scores',...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_obs-offset-1*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, 'scores'));
            uicontrol(ctl, ...
                'String', 'SPE',...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_obs-offset-2*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, 'SPE'));
            
            uicontrol(ctl, ...
                'String', 'Predictions',...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_obs-offset-3*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, 'predictions'));
            
            % Variable-based plots
            % -----------------------
            nbuttons = 2;
            h_var = nbuttons*(bheight + delta) + offset;
            ctl.Parent = hQuick;
            pVar = uipanel(ctl, 'Title', 'Variable-based plots', ...
                'Position', [delta H-h_obs-delta-h_var w h_var]);
            
            ctl.Parent = pVar;
            uicontrol(ctl, ...
                'String', 'Loadings', ...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_var-offset-1*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, 'loadings'));
            
            uicontrol(ctl, ...
                'String', 'VIP plots',...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_var-offset-2*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, 'VIP'));
            
            % Component-based plots
            % -----------------------
            nbuttons = 1;
            h_cmp = nbuttons*(bheight + delta) + offset;
            ctl.Parent = hQuick;
            pCmp = uipanel(ctl, 'Title', 'Component-based plots', ...
                'Position', [delta H-h_obs-delta-h_var-delta-h_cmp w h_cmp]);
            
            ctl.Parent = pCmp;
            uicontrol(ctl, ...
                'String', 'R2', ...
                'Style', 'Pushbutton', ...
                'Position', [delta, h_cmp-offset-1*(bheight+delta)+delta, w-2*delta, bheight], ...
                'Callback', @(src,event)plot(self.model, 'R2'));
            
           
            
            set(hQuick, 'Visible' ,'on')
        end
        
        function h = new_figure(self, varargin)
            % Syntax: self.new_figure('off')
            % Creates a new figure; set the visibility for it
            % (Visibility off is useful for unit testing)
            
            self.hF = figure(...
                 'Visible',     'on', ...
                 'DoubleBuffer',  'on', ...  % Allows fast redrawing of images
                 'BackingStore', 'off', ...  % Faster animation when figures are updated frequently
                 'WindowButtonDownFcn', @mouseclick_callback);

            set(self.hF, 'Color', self.opt.fig.backgd_colour);
            if self.opt.fig.add_menu
                set(self.hF, 'ToolBar', 'figure');
            end
            units = get(self.hF, 'Units');
            set(self.hF, 'units', 'Pixels');
             
            fPos = get(self.hF, 'position');
            fPos(1) = round(0.05*self.screen(3));
            fPos(2) = round(0.1*self.screen(4));
            fPos(3) = 0.90*self.screen(3);
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
            self.hFoot = uicontrol(Txt, ...
                             'Position',[0.02, 0.005, 0.96 0.04], ...
                             'ForegroundColor',[0 0 0], 'String', '', ...
                             'HorizontalAlignment', 'left');
                         
                         
            % Add a contribution toolbar icon
            icon = fullfile(matlabroot,'/toolbox/matlab/icons/greenarrowicon.gif');
            [cdata,map] = imread(icon);
 
            % Convert white pixels into a transparent background
            map(find(map(:,1)+map(:,2)+map(:,3)==3)) = NaN;
 
            % Convert into 3D RGB-space
            cdataRedo = ind2rgb(cdata,map);
            % Add the icon (and its mirror image = undo) to the latest toolbar
            hToolbar = findall(self.hF,'tag','FigureToolBar');
            hSimulate = uipushtool('Parent', hToolbar, 'cdata',cdataRedo, 'tooltip','undo', 'ClickedCallback','uiundo(gcbf,''execUndo'')');
            set(hSimulate,'Separator','on');

                         
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
                ctl.FontSize = 10;
                self.hM(i) = uicontrol(ctl, ...
                    'String', 'AA', ...
                    'ForegroundColor',[0 0 0], ...
                    'Position',[hPos_pix(1)-50, hPos_pix(2)-25, 15 15], ...
                    'TooltipString', 'Adjust axes', ...
                    'Callback', @adjust_axes);
                set(self.hM(i), 'Units', 'Normalized')
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
        
        

        
        
    end % end: methods (ordinary)
    
    methods (Static=true)
        
    end % end: methods (static)
end % end: ``classdef``

function mouseclick_callback(varargin)
    disp(get(varargin{1}, 'UserData'))
end

function adjust_axes(varargin)
    hOb = varargin{1};
    self = get(get(hOb, 'Parent'), 'UserData');
    
    % User wants to change axes
    if get(hOb, 'Value') == 1
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
       
        hFr = uipanel(ctl, 'Title', 'X-axis series', ...
            'Position', [H W H/0.95*0.10 H/0.95*0.35], ...
            'BorderType', 'beveledout', ...
            'TitlePosition', 'centertop');
        set(hFr, 'Units', 'pixels')
        hFr_pos = get(hFr, 'Position');
        set(hFr, 'Position', [hFr_pos(1) hFr_pos(3), 200, 50])
        self.hSelectX = hFr;
        
        %set(self.hF, 'Units', 'Normalized');
        ctl.Units = 'pixels';
        hFr = uipanel(ctl, 'Title', 'Y-axis series', ...
            'Position',[hFr_pos(4) hFr_pos(2), 200, 50], ...
            'BorderType', 'beveledout', ...
            'TitlePosition', 'centertop');
        
        self.hSelectY = hFr;
        
        set(self.hF, 'Units', units);
        
        
    % User wants to finish axis change
    elseif get(hOb, 'Value') == 0
        if ishandle(self.hSelectX)
            delete(self.hSelectX);
        end
        if ishandle(self.hSelectY)
            delete(self.hSelectY);
        end
    end
end