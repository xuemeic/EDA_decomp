%% test: for n_trials = 40 and max_K=4, took about 10min

n_trials = 40;
max_K = 4;

% initialization
clear T error
T.cs1 = zeros(max_K, n_trials);
error.cs1 = T.cs1;

T.cs2 = T.cs1;
error.cs2 = T.cs1;

T.gms = T.cs1;
error.gms = T.cs1;

for K = 1:max_K
for i = 1:n_trials
    [cs1, cs2, gms] = test_par_recovery(K, 5, 0.8);
    T.cs1(K, i) = cs1.time;
    error.cs1(K, i) = cs1.error;

    T.cs2(K, i) = cs2.time;
    error.cs2(K, i) = cs2.error;

    T.gms(K, i) = gms.time;
    error.gms(K, i) = gms.error;
end
end

% 5 and 0.8 seems to be a good choice
%save("test_par_5_p8.mat","T","error")
%% see results
%load('test_par_5_p8.mat')
time_t = table;

time_t.cs1 = mean(T.cs1, 2);
time_t.cs2 = mean(T.cs2, 2);
time_t.gms = mean(T.gms, 2);
%time_t.Properties.RowNames = {'K=2', 'K=4'};
time_t

error_t = time_t;
error_t.cs1 = mean(error.cs1, 2);
error_t.cs2 = mean(error.cs2, 2);
error_t.gms = mean(error.gms, 2);

error_t