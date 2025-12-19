% 12/19/2025

%% Create synthetic data
input.n = 370;
input.X_sparsity = 10;
input.X_supp = 'unif';
input.delta = 0.1; %0.4
input.gamma = 0.1; %0.4
input.epsilon = 0.2; %0.1
input.K = 40;
input.alpha = 1;
input.B_mode = 'jump';
input.DB_sparsity = 2;
input.tau1 = 2;
input.tau2 = 0.75;
input.H_mode = 'tril';


oo = create_synthetic_data(input);
N = size(oo.Y, 1); % same as input.T
K = input.K;

cs_method = 'lasso';
%% Compressed Sensing Method
%DH_I = [oo.D*oo.H, eye(N-1)];
% Create matrix to store recovered values
T = size(oo.X, 1);
%recovered_X = zeros(T, input.K);

A = [oo.D*oo.H, eye(N-1)];
% Time the compressed sensing method


% Iterate through columns to apply Compressed Sensing method
tic


if strcmp(cs_method, 'cvx')
    recovered_X = compressed_sensing(oo.Y, oo.D, oo.H, oo.eta);
elseif strcmp(cs_method, 'lasso')
    paral.method = 'FISTA';
    paral.tol = 1e-5;
    paral.max_iter = 200;
    l_lam = 0.02;
    [recovered_Z, ~] = my_lasso(A, -diff(oo.Y), l_lam, paral);
    recovered_X = recovered_Z(1:T,:);
end


compressed_sensing_time = toc

% Calculate the error for the Compressed Sensing method
%x_error = norm(recovered_X - oo.X, 'fro')/norm(oo.X, 'fro')
x_error_vec = sqrt(sum((recovered_X - oo.X).^2, 1))./sqrt(sum((oo.X).^2, 1));
mean(x_error_vec)

%% GMS method

% Set parameters
para.rho_outer = 1; 
para.lasso_rho = 1; 
para.max_iter = 100;
para.lasso_max_iter = 50;
para.tol_outer = 1e-8;
para.lasso_tol = 1e-5;
para.lasso_decomp = "svd";      
para.lasso_method = 'FISTA';

lam =  1/sqrt(max(N, K))*3; % 1/sqrt(size(Y,1)) for best results

% Time the Matrix Separation method
tic
output = gen_matrix_sep_con(oo.Y, oo.H, lam, para);
matrix_separation_time = toc

% Error for Matrix Separation

%rel_L = norm(output.L - oo.B, 'fro')/norm(oo.B, 'fro') % Baseline error
x_error_vec2 = sqrt(sum((output.S - oo.X).^2, 1))./sqrt(sum((oo.X).^2, 1));
mean(x_error_vec2)



%% Plot
figure(1)
ii = 2;
subplot(2,1,1)
plot(1:T, oo.X(:,ii), 1:T, output.S(:,ii))
legend('truth','matrix')
subplot(2,1,2)
plot(1:T, oo.X(:,ii), 1:T, recovered_X(:,ii))
legend('truth', 'cs')

