function x_recovered = compressed_sensing(y, D, H, eta)
%
% y := raw signal (column vector)
% DH_I := matrix for optimization DH_I = [DH I]
% eta := noise tolerance
%
%%%%%%%%%%%%%

% Define arguments
arguments
    y
    D
    H
    eta = 1e-1
end


% Clear any previous CVX problem
clear cvx;

[N, T] = size(H);

A = [D*H, eye(N-1)];

z_dim = T + N - 1;

K = size(y, 2);

% y is N by K
dy = -1*diff(y); 
% dy is (N-1) by K

cvx_begin quiet
    variable z(z_dim, K)
    minimize(sum(abs(z(:))))                % Promote sparsity in z
    subject to
        norm(dy - A*z, 'fro') <= sqrt(K)*eta;    % Data noise constraint
        z(1:T,:) >= 0;                    % Require positivity on recovered signal
cvx_end

% Store the recovered signal
x_recovered = z(1:T, :);

end
