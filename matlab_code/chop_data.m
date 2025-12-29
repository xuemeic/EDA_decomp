function Y = chop_data(data, num_of_chops, overlap)
% chop_data slices each column of input into segments with optional overlap
% This is processed per column, as different col represents different participant.
% Inputs:
% - data := input matrix 
% - num_of_chops := number of segments per column if overlap = 0
% - overlap := overlap ratio, at most 0.5
%
% Returns:
% - Y: matrix of chopped data 
% rewritten on 8/13/2025 as overlap was not working properly

arguments
    data
    num_of_chops = 2
    overlap = 0
end

[num_rows, num_cols] = size(data);

% num_of_chops will determine the length of columns in Y
Y_nrows = ceil(num_rows/num_of_chops);

overlap_length = floor(Y_nrows * overlap);
nonlap_length = Y_nrows - overlap_length;
temp = floor((num_rows - Y_nrows)/nonlap_length);
if temp < (num_rows - Y_nrows)/nonlap_length
    multiplier = temp + 2;
else
    multiplier = temp + 1;
end

Y = zeros(Y_nrows, num_cols*multiplier);
for col = 1:num_cols
    y = zeros(Y_nrows, multiplier);
    if temp < (num_rows - Y_nrows)/nonlap_length
        for i = 1:(multiplier-1)
            start_idx = nonlap_length*(i-1)+1;
            end_idx = nonlap_length*(i-1)+ Y_nrows;
            y(:,i) = data(start_idx:end_idx, col);
        end
        % last col of y is different
        y(:,multiplier) = data((num_rows-Y_nrows+1):num_rows, col);
    else
        for i = 1:(multiplier)
            start_idx = nonlap_length*(i-1)+1;
            end_idx = nonlap_length*(i-1)+ Y_nrows;
            y(:,i) = data(start_idx:end_idx, col);
        end
    end
    s_idx = (col-1)*multiplier+1;
    e_idx = (col)*multiplier;
    Y(:, s_idx:e_idx) = y;

   
    
end

end
