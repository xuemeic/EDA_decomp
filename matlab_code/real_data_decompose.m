function X = real_data_decompose(eda, n_chops, overlap, n_participants)
%%% eda: each column is one participant's data, typically 1360 by 27
%%% n_participants: a value between 1 and 27. It will bundle
% n_participants together to process them in one batch. for example,
%%% output X is recovered SCR events

X = zeros(size(eda));
ncols_eda = size(eda, 2);



%num_of_chops = 5;
%overlap = 0.5;


n_batch = floor(ncols_eda/n_participants); % 9

tau1 = 2; % default is 10
tau2 = 0.75; % default is 1

Y = chop_data(eda(:,((1-1)*n_participants + 1):(1*n_participants)), n_chops, overlap);
[N, K] = size(Y);
H = make_H(N, tau1, tau2, 'tril');
lam =  1/sqrt(max(N, K))*3;
para.rho_outer = 1; 
para.lasso_rho = 1; 
para.max_iter = 100;
para.lasso_max_iter = 50;
para.tol_outer = 1e-8;
para.lasso_tol = 1e-5;
para.lasso_decomp = "svd";      
para.lasso_method = 'FISTA';
for i = 1:n_batch
    eda_c = eda(:,((i-1)*n_participants + 1):(i*n_participants));
    Y = chop_data(eda_c, n_chops, overlap);
    

% decompose via GMS

output = gen_matrix_sep_con(Y, H, lam, para);
X_gms = reconstr_data(output.S, size(eda_c), n_chops, overlap);
X(:,((i-1)*n_participants + 1):(i*n_participants)) = X_gms;
end
end