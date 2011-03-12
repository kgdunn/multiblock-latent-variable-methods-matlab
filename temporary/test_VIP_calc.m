clean
X_raw = [3, 4, 2, 2; 4, 3, 4, 3; 5.0, 5, 6, 4];
[N,K] = size(X_raw);
X = block(X_raw);
[data, PP] = preprocess(X);  % or X = X.preprocess();

X = data.data;
SS_0_rem = ssq(X, 1);

P_1 = [0.5410, 0.3493, 0.5410, 0.5410]';
P_2 = [-0.201689, 0.9370, -0.201689, -0.201689]';
P = [P_1 P_2];
T_1 = [-1.6229, -0.3493, 1.9723]';
T_2 = [0.6051, -0.9370, 0.3319]';



X_hat_1 = T_1 * P_1';
X = X - X_hat_1;
SS_1_exp = ssq(X_hat_1, 1);
SS_1_rem = ssq(X, 1);
SS_1_cum = SS_0_rem - SS_1_rem;

X_hat_2 = T_2 * P_2';
X = X - X_hat_2;
SS_2_exp = ssq(X_hat_2, 1);
SS_2_rem = ssq(X, 1);
SS_2_cum = SS_0_rem - SS_2_rem;


% Main literature form: fraction of total sum of squares explained:
% After 1 PC
SS_frac_1 = sum(SS_1_exp)/sum(SS_1_cum);
VIP_1 = sqrt(P_1'.^2 .* SS_frac_1 .* K)  % agrees

% After 2 PCs
SS_frac_1 = sum(SS_1_exp)/sum(SS_2_cum);
SS_frac_2 = sum(SS_2_exp)/sum(SS_2_cum);
VIP_2 = sqrt(P_1'.^2 .* SS_frac_1 .* K  + P_2'.^2 .* SS_frac_2 .* K )
