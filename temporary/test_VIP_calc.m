clean
addpath('../')


dataset = '3points';

if strcmp(dataset, '4points')
    X_raw = [3, 4, 2, 2; 4, 3, 4, 3; 5.0, 5, 6, 4; 5.0, 5, 6, 3];
    [N,K] = size(X_raw);
    X = block(X_raw);
    [data, PP] = preprocess(X); 
    P = [0.545437378946789	-0.109587134692576;0.403543519066049	0.86908013987617;0.545437378946789	-0.109587134692576;0.492086130179568	-0.469766995746359];
    T = [-2.1322787324264	0.634563634058212;-0.81170470962319	-1.07742574176906;1.77333170196907	-0.0662413007261797;1.17065174008051	0.509103408437029];
elseif strcmp(dataset, '3points')
    X_raw = [3, 4, 2, 2; 4, 3, 4, 3; 5.0, 5, 6, 4];
    [N,K] = size(X_raw);
    X = block(X_raw);
    [data, PP] = preprocess(X); 
    
    P_1 = [0.5410, 0.3493, 0.5410, 0.5410]';
    P_2 = [-0.201689, 0.9370, -0.201689, -0.201689]';
    P = [P_1 P_2];
    T_1 = [-1.6229, -0.3493, 1.9723]';
    T_2 = [0.6051, -0.9370, 0.3319]';
    T = [T_1 T_2];
end
X = data.data;
SS_0_rem = ssq(X, 1);



%X_raw = [3, 4, 2, 2; 4, 3, 4, 3; 5.0, 5, 6, 4; 5.0, 5, 6, 3];
P_1 = P(:,1);
P_2 = P(:,2);
T_1 = T(:,1);
T_2 = T(:,2);

% Hotelling's T2
HT2 = T_1 .^ 2 ./ var(T_1);
HT2_cov = diag(T_1 * inv((T_1'*T_1)/(N-1)) * T_1');
HT2 - HT2_cov

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
