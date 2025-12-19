%%%%%% choose parameters %%%%%%
% choose 'exp' or 'unif'
% if 'exp', then entries of support is exponentially distributed
% if 'unif', then entries of support is uniformly distributed
input.X_supp = 'unif'; 

% choose 'mild' or 'jump'
% if 'mild', baseline is slow varying
% if 'jump', baseline has jumps
input.B_mode = 'mild'; 

% choose 'cvx' or 'lasso'
%%%%%%%%%% WARNING: if 'cvx', takes 14-20 min, slightly more accurate %%%%%%%%%%%
cs_method = 'lasso'; % a lot faster!

% 
n = 240;
K = 40;
sparsity_range = 1:3:31;
epsilon_range = [0.02, 0.08, 0.16, 0.32, 0.64, 1];

nrows = length(epsilon_range);
ncols = length(sparsity_range);
cs_results = zeros([nrows, ncols]);
gms_results = cs_results;

input.n = n;
input.DB_sparsity = 1;


input.delta = 0.01; %0.4
input.gamma = 0.01; %0.4
input.K = K;
input.alpha = 1;

input.tau1 = 2;
input.tau2 = 0.75;
input.H_mode = 'tril';

rng(1)
oo = create_synthetic_data(input);
A = [oo.D*oo.H, eye(n-1)];

% parameters for CS method
paral.method = 'FISTA';
paral.tol = 1e-5;
paral.max_iter = 200;
l_lam = 1e-2;


% parameters for GMS method
para.rho_outer = 1; 
para.lasso_rho = 1; 
para.max_iter = 100;
para.lasso_max_iter = 50;
para.tol_outer = 1e-8;
para.lasso_tol = 1e-5;
para.lasso_decomp = "svd";      
para.lasso_method = 'FISTA';
lam =  1/sqrt(max(n, K))*3;
for i = 1:nrows
    input.epsilon = epsilon_range(i);
    i
    for j = 1:ncols
        
        input.X_sparsity = sparsity_range(j);
        oo = create_synthetic_data(input);
        % CS method
        if strcmp(cs_method, 'cvx')
        recovered_X = compressed_sensing(oo.Y, oo.D, oo.H, oo.eta);
        recovered(i,j).cs = recovered_X;
        else
        [recovered_Z, ~] = my_lasso(A, -diff(oo.Y), l_lam, paral);
        recovered_X = recovered_Z(1:n,:);
        end
        x_error_vec = sqrt(sum((recovered_X - oo.X).^2, 1))./sqrt(sum((oo.X).^2, 1));
        cs_results(i,j) = mean(x_error_vec);

        % GMS method
        output = gen_matrix_sep_con(oo.Y, oo.H, lam, para);
        x_error_vec2 = sqrt(sum((output.S - oo.X).^2, 1))./sqrt(sum((oo.X).^2, 1));
        gms_results(i,j) = mean(x_error_vec2);
        recovered(i,j).gms = output.S;
        recovered(i,j).truth = oo.X;


    end
end

%% plot

figure(1);
subplot(1,2,1)
imagesc(cs_results, 'XData', sparsity_range);
colorbar;
set(gca, 'YDir', 'normal');
xticks = sparsity_range;
set(gca, 'XTick', xticks, 'XTickLabel', xticks);
yticks = epsilon_range;
set(gca, 'YTickLabel', yticks);
clim([0,.4]);
xl = xlabel('number of SCR events s');
yl = ylabel('noise level \epsilon');
xl.FontSize = 15;
yl.FontSize = 15;
t1 = title("Relative Error via CS method");
t1.FontSize = 16;

subplot(1,2,2)
imagesc(gms_results, 'XData', sparsity_range);
colorbar;
set(gca, 'YDir', 'normal');
colorbar;
set(gca, 'YDir', 'normal');
xticks = sparsity_range;
set(gca, 'XTick', xticks, 'XTickLabel', xticks);
yticks = epsilon_range;
set(gca, 'YTickLabel', yticks);
clim([0,.4]);
xl = xlabel('number of SCR events s');
yl = ylabel('noise level \epsilon');
xl.FontSize = 15;
yl.FontSize = 15;
t1 = title("Relative Error via GMS method");
t1.FontSize = 16;


