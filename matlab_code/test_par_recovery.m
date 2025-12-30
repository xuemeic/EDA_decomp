%% Test on parallelled recovery
% 12/25/2025
function [cs1, cs2, gms] = test_par_recovery(K, n_chops, overlap)
%% Create synthetic data
%rng(3)
input.n = 1360; 
input.X_sparsity = 20;
input.X_supp = 'unif';
input.delta = 10; 
input.gamma = 10; 
input.epsilon = 0.3; 
input.K = 1;
input.alpha = 1;
input.B_mode = "mild";
input.DB_sparsity = 1;
input.tau1 = 2;
input.tau2 = 0.75;
input.H_mode = 'tril';

oo1 = create_synthetic_data(input);
oo = oo1;
if K > 1
    
    for j = 1:(K-1)
        input2 = input;
        input2.X_supp = 'unif';
        input2.B_mode = 'mild';
        input2.DB_sparsity = 1;
        oo2 = create_synthetic_data(input2);
        oo.Y = [oo.Y, oo2.Y];
        oo.X = [oo.X, oo2.X];
    end
end
%{
input2 = input;
input2.X_supp = 'unif';
input2.B_mode = "jump";
input2.DB_sparsity = 1;
oo2 = create_synthetic_data(input2);

oo = oo1;
oo.Y = [oo1.Y, oo2.Y]; oo.X = [oo1.X, oo2.X];
%}
cs_method = 'lasso';

%% reshape data
% 5, 0.8
%n_chops = 5;
%overlap = 0.7;
Y_chopped = chop_data(oo.Y, n_chops, overlap);

%% Compressed Sensing Method 1: traditional, process the whole signal
N = input.n; 
A1 = [oo.D*oo.H, eye(N-1)];

tic
if strcmp(cs_method, 'cvx')
    recovered_X1 = compressed_sensing(oo.Y, oo.D, oo.H, oo.eta);
elseif strcmp(cs_method, 'lasso')
    paral.method = 'FISTA';
    paral.tol = 1e-5;
    paral.max_iter = 200;
    l_lam = 0.02;
    [recovered_Z, ~] = my_lasso(A1, -diff(oo.Y), l_lam, paral);
    recovered_X1 = recovered_Z(1:N,:);
    recovered_X1(recovered_X1<0)=0;
end
cs1_time = toc;

% Calculate the relative error for the Compressed Sensing method
cs1_error = calc_error(recovered_X1, oo.X);
%fprintf('cs1 error is %.3f; runtime is %.3f.\n', cs1_error, cs1_time)

%% Compressed Sensing Method 2: parallelled recovery

n1 = size(Y_chopped, 1); % new signal length
n2 = size(Y_chopped, 2);

paral.method = 'FISTA';
paral.tol = 1e-5;
paral.max_iter = 200;
l_lam = 1e-2;
D = make_D(n1);
H = make_H(n1, input.tau1, input.tau2, input.H_mode);
A2 = [D*H, eye(n1-1)];
tic
[recovered_Z, ~] = my_lasso(A2, -diff(Y_chopped), l_lam, paral);
recovered_X2 = recovered_Z(1:n1,:);


X_cs2 = reconstr_data(recovered_X2, size(oo.Y), n_chops, overlap);
X_cs2(X_cs2<0)=0;
cs2_time = toc;

cs2_error = calc_error(X_cs2, oo.X);
%fprintf('cs2 error is %.3f; runtime is %.3f.\n', cs2_error, cs2_time)

%% GMS
para.rho_outer = 1; 
para.lasso_rho = 1; 
para.max_iter = 100;
para.lasso_max_iter = 50;
para.tol_outer = 1e-8;
para.lasso_tol = 1e-5;
para.lasso_decomp = "svd";      
para.lasso_method = 'FISTA';


lam =  1/sqrt(max(n1, n2))*5; 

% Time the Matrix Separation method
tic
output = gen_matrix_sep_con(Y_chopped, H, lam, para);
X_gms = reconstr_data(output.S, size(oo.Y), n_chops, overlap);
gms_time = toc;
X_gms(X_gms<0)=0;
% Error for Matrix Separation
gms_error = calc_error(X_gms, oo.X);
%calc_support_error(output.S, oo.X, 0.05)
%fprintf('gms error is %.3f; runtime is %.3f.\n', gms_error, gms_time)

cs1.time = cs1_time;
cs1.error = cs1_error;

cs2.time = cs2_time;
cs2.error = cs2_error;

gms.time = gms_time;
gms.error = gms_error;
end

function me = calc_error(approx, truth)
% input approx and truth will have the same size
% computes relative error, averaged over all columns
error_vec = sqrt(sum((approx - truth).^2, 1))./sqrt(sum((truth).^2, 1));
me = mean(error_vec);
end

function D = make_D(M)
% return (M-1) by M D = [1 -1 0 ...]
d_row = zeros(M, 1);
d_col = zeros(1, M-1);

d_row(1:2) = [1 -1];
d_col(1) = 1;

D = toeplitz(d_col, d_row);
end