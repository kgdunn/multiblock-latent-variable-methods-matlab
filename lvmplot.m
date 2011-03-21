% Copyright (c) 2010-2011 ConnectMV, Inc. All rights reserved.
% -------------------------------------------------------------------------
%
% Code for plotting and interrogating latent variable models.


% Create a handle class
% This is the highest level latent variable model class

classdef lvmplot < handle
    properties
        hF = 0;             % Figure window
        hA = [];            % Matrix of axes [nRow by nCol]
        nA = 0;             % Number of axes in plot
        nRow = 0;           % Number of rows of axes
        nCol = 0;           % Number of columns of axes
        model = [];         % The model being plotted
        
        hFoot = 0;          % Handle to the footer
        % Default plotting options
        opt = struct;
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
            self.new_figure();
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
        function h = get.hF(self)
            h = self.hF;
        end
        
        function h = get.hA(self)
            h = self.hA(:);
        end
        
        function h = new_figure(self, varargin)
            % Syntax: self.new_figure('off')
            % Creates a new figure; set the visibility for it
            % (Visibility off is useful for unit testing)
            
            
            self.hF = figure('Visible', 'off');
            set(self.hF, 'Color', self.opt.fig.backgd_colour);
            if self.opt.fig.add_menu
                set(self.hF, 'ToolBar', 'figure');
            end
            units = get(self.hF, 'Units');
            set(self.hF, 'units', 'Pixels');
             
            screen = get(0,'ScreenSize');   
            fPos = get(self.hF, 'position');
            fPos(1) = round(0.05*screen(3));
            fPos(2) = round(0.1*screen(4));
            fPos(3) = 0.90*screen(3);
            fPos(4) = 0.80*screen(4);
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
                         
            set(self.hF, 'units', units);
            if nargin > 1
                set(self.hF, 'Visible', varargin{1});
            else
                set(self.hF, 'Visible', self.opt.fig.default_visibility)
            end
            h = self.hF;
        end
        
    end % end: methods (ordinary)
end % end: ``classdef``
