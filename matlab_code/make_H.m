function H = make_H(n, tau1, tau2, mode)
%%%%%%%%%% Inputs 
% n: signal length, need to be bigger than 160
% tau1 > tau2 > 0

%%%%%%%%%% output 
% H: n by n matrix

% Using model from Compressed Sensing paper, construct H
% updated 12/18/2025
% Xuemei Chen

T = 40; % duration in seconds
scale = 1;
TT = T/scale;
fs = 4; % signal sampled at every 1/fs second
f_len = TT*fs;
u = linspace(0, TT, f_len);

uu = u*scale;
f = 2 * ((exp(-uu / tau1)) - (exp(-uu / tau2))); % 160 coordinates

ncols_main_H = n + f_len - 1;
h = zeros(n + f_len - 1, 1);
h(1:f_len) = f;

% Construct the Toeplitz matrix
hflip = fliplr(h');
r = circshift(hflip, 1);
H_temp = toeplitz(h, r); 

main_H = H_temp(:,1:n); % n+f_len-1 by n

switch mode
    case 'top'
        H = main_H(1:n,:);
    case 'tril'
        H = main_H(1:n,:);
        H = tril(H);
    case 'middle'
        center_idx = floor(ncols_main_H/2);
        half_n = floor(n/2);
        H = main_H((center_idx - half_n):(n + center_idx - half_n -1), :);
end

end