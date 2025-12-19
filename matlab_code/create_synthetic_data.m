function output = create_synthetic_data(input)
%%%%%%%%%%%%%%%%%%%%% Inputs
% .tau1
% .tau2
% .H_mode
% .n: signal length
% .K: number of signals created
% .B_mode: baseline model mode. if 'mild', there is no jump; if 'paper', there is jump
% .X_sparsity: number of events of each signal
% .X_supp: if 'exp', entry follows exponential distribution; if 'unif',
% entry follows uniform distribution
% .DB_sparsity: denoted as c. only activated if B_mode is 'paper'
% .gamma: ||Db - Db_c||_1 <= gamma
% .delta: ||x - x_s||_1 <= delta
% .epsilon: noise level in each signal as ||e||_2<= epsilon
%%%%%%%%%%%%%%%%%%%%% outputs
%
% 
%%%%%%%%%%%%%%%%%%%%%%
% 12/19/2025
% Xuemei Chen


X_sparsity = input.X_sparsity;
DB_sparsity = input.DB_sparsity;
delta = input.delta;
gamma = input.gamma;
epsilon = input.epsilon;
K = input.K;
n = input.n;
alpha = input.alpha;
eta = 1.05*epsilon;

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Impulse Response Matrix (H)
H = make_H(n, input.tau1, input.tau2, input.H_mode);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Creating D matrix: (n-1) x n
% D = [1, -1, 0, ..., 0;
%      0, 1, -1, ..., 0;
%      ..., ...        ;
%      0, 0, ..., 1, -1]

d_row = zeros(n, 1);
d_col = zeros(1, n-1);
d_row(1:2) = [1 -1];
d_col(1) = 1;
D = toeplitz(d_col, d_row);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating baseline variation (B) as an n × K matrix with each column being
% a baseline signal

if strcmp(input.B_mode, 'mild')
B = zeros(n, K);

for i = 1:K
    % Random Baseline
    l = randi([1, 4]);

    % Interpolation spline
    number_nodes = 4;
    x = linspace(1, n, number_nodes);
    y = [l, l+0.2, l+0.4, l+0.3];
    cs = spline(x, [0 y 0]);

    B(:, i) = ppval(cs, 1:n)';  
end
DB = D*B;
for i = 1:K
    %num_events = randi([1, ceil(N/10)]);  % Random number of activations per event
    
    xe = randn([n-1,1]);
    xe = gamma*xe/sum(abs(xe));
    DB(:, i) = DB(:, i) + xe;
end
temp = cumsum(DB, 1);
temp = -temp;
temp_first_row = B(1,:);
temp = temp + temp_first_row; % each row of B is being added by B_first_row;
temp = [temp_first_row; temp];
B = temp;

elseif strcmp(input.B_mode, 'jump')
    % create DB first
    DB = zeros(n-1, K);
    c = DB_sparsity;
    for i = 1:K
    %num_events = randi([1, ceil(N/10)]);  % Random number of activations per event
    x = zeros(n-1, 1);
    support = randperm(n-1, c);  % Pick multiple positions
    
    x(support) = randn([length(support), 1]);
    xe = randn(size(x));
    xe = gamma*xe/sum(abs(xe));
    DB(:, i) = x + xe;
    end
% each col of B is [b1-b2, b1-b3, b1-b4, ..., b1-b_N];
    B = cumsum(DB, 1); % N-1 by K

% each col of B is [-b1+b2, -b1+b3, -b1+b4, ..., -b1+b_N];
    B = -B;

%B_first_row = exprnd(2, [1, K]);
B_first_row = randn([1, K]) + 4;
B = B + B_first_row; %each row of B is being added by B_first_row;
B = [B_first_row; B];
end

%%%%%%%%%%%%%%%%%%%%% SCR Event Matrix (X) of size n × k
X = zeros(n, K);
s = X_sparsity;
for i = 1:K
    %num_events = randi([1, ceil(N/10)]);  % Random number of activations per event
    x = zeros(n, 1);
    support = randperm(n, s);  % Pick multiple positions
    if strcmp(input.X_supp, 'exp')
        % each entry of x on the support follows exponential distribution with
        % meanf = 2
        x(support) = exprnd(2, [length(support), 1]);
    elseif strcmp(input.X_supp, 'unif')
        x(support) = 2 + (7-2)*rand([length(support), 1]);
    end
    xe = randn(size(x));
    xe = delta*xe/sum(abs(xe));
    X(:, i) = x + xe;
end

%%%%%%%%%%%%%%%%%%%%%
% SCR Response (R) and Noise (E)
R = H * X;  % Each column in R corresponds to one event response

% Generate noise E
E = randn(n, K);
E = epsilon*normc(E);

%%%%%%%%%%%%%%%%
% Generate the synthetic EDA signal (Y)
Y = R + alpha*B + E;

% Construct the DH_I matrix for comp_sensing function
%DH_I = [D * H, eye(N - 1)];

output.D = D;
output.H = H;
output.Y = Y;
output.X = X;
output.DB = DB;
output.B = B;
output.eta = eta;
end
