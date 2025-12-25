% 12/19/2025

%% Create synthetic data
rng(3)
input.n = 370;
input.X_sparsity = 10;
input.X_supp = 'unif';
input.delta = 10; %0.4
input.gamma = 10; %0.4
input.epsilon = 0.3; %0.1
input.K = 40;
input.alpha = 1;
input.B_mode = 'jump';
input.DB_sparsity = 1;
input.tau1 = 2;
input.tau2 = 0.75;
input.H_mode = 'tril';

oo = create_synthetic_data(input);
N = size(oo.Y, 1); 
K = input.K;

cs_method = 'lasso';
%% Compressed Sensing Method

n = size(oo.X, 1);
A = [oo.D*oo.H, eye(N-1)];

tic
if strcmp(cs_method, 'cvx')
    recovered_X = compressed_sensing(oo.Y, oo.D, oo.H, oo.eta);
elseif strcmp(cs_method, 'lasso')
    paral.method = 'FISTA';
    paral.tol = 1e-5;
    paral.max_iter = 200;
    l_lam = 0.02;
    [recovered_Z, ~] = my_lasso(A, -diff(oo.Y), l_lam, paral);
    recovered_X = recovered_Z(1:n,:);
end
cs_time = toc;

% Calculate the relative error for the Compressed Sensing method
cs_error = calc_error(recovered_X, oo.X);
fprintf('cs error is %.3f; runtime is %.3f.\n', cs_error, cs_time)

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
gms_time = toc;

% Error for Matrix Separation
gms_error = calc_error(output.S, oo.X);
%calc_support_error(output.S, oo.X, 0.05)
fprintf('gms error is %.3f; runtime is %.3f.\n', gms_error, gms_time)

%% Plot
figure(1)
ii = 5;
subplot(2,1,1)
%plot(1:n, oo.X(:,ii), 1:n, output.S(:,ii))
plot1sig(output.S(:,ii), oo.X(:,ii), 0.2, 'GMS');

subplot(2,1,2)
plot1sig(recovered_X(:,ii), oo.X(:,ii), 0.2, 'CS');

%% print plot
%print("-f1", '../plots/synthetic_initial', '-djpeg', '-r500')
function plot1sig(sig, t, alpha, method)
% sig: recovered
% t: ground truth
support = find(abs(t) >= alpha);
stem(support, t(support))
hold on
plot(sig);
xlim([0,370])
legend('ground truth', 'recovered', 'Location', 'best')
tl = title(sprintf('Recovery of x by the %s method', method));
tl.FontSize = 16;
hold off
end

function me = calc_error(approx, truth)
% input approx and truth will have the same size
% computes relative error, averaged over all columns
error_vec = sqrt(sum((approx - truth).^2, 1))./sqrt(sum((truth).^2, 1));
me = mean(error_vec);
end

function me = calc_support_error(approx, truth, alpha)
truth(abs(truth)<alpha) = 0;
truth(abs(truth)>=alpha) = 1;
approx(abs(approx)<alpha) = 0;
approx(abs(approx)>=alpha) = 1;
me = calc_error(approx, truth);
end

