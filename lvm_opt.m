function opt = lvm_opt(varargin)
% Returns options for all latent variable model types.

opt.md_method = 'scp';
opt.max_iter = 15000;
opt.show_progress = true;         % Show a progress bar with option to cancel
opt.min_lv = -1;
opt.max_lv = inf;
opt.build_now = true;
opt.tolerance = sqrt(eps);
opt.stop_now = false;             % Periodically checked; used to stop any loops

opt.mbpls.block_scale_X = true;                  % The various X-blocks should be block scaled
opt.mbpls.deflate_X = true;                      % No need to deflate the X-blocks, but, we must account for it in the PLS model


opt.batch.calculate_monitoring_limits = false;   % Calculate monitoring limits for batch blocks
opt.batch.monitoring_level = 0.95;
opt.batch.monitoring_limits_show_progress = true;

opt.randomize_test = struct;
opt.randomize_test.use = false;
opt.randomize_test.points = 0;                    % Start with zero points
opt.randomize_test.risk_uncertainty = [0.5 10.0]; % Between these levels the percentage risk is considered uncertain, and could be due 
                                                 % to the  randomization values.  So will do more permutationsm, to a maximum of 3 
                                                 % times the default amount, to more clearly define the risk level.
opt.randomize_test.permutations = 500;           % Default number of permutations.
opt.randomize_test.max_rounds = 20;              % We will do at most 1000 permutations to assess risk
opt.randomize_test.test_statistic = [];
opt.randomize_test.risk_statistics = cell(1,1);
opt.randomize_test.last_worthwhile_A = 0;        % The last worthwhile component added to the model
opt.randomize_test.show_progress = true;
opt.randomize_test.temp_data = {};               % Temporary data during the randomization routine


% Cross-validation is not working as intended.
% 
% opt.cross_val = struct;
% opt.cross_val.use = false;  % use cross-validation or not
% opt.cross_val.groups = 5;   % not of groups to use
% opt.cross_val.start_at = 4; % sequential counting of groups starts here
% opt.cross_val.strikes = 0;  % number of strikes encountered already
% opt.cross_val.PRESS = [];
% opt.cross_val.PRESS_0 = [];


