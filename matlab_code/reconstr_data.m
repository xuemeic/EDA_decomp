function data = reconstr_data(Y, data_size, num_of_chops, overlap)
% this is the reverse of chop_data()
% take the average at overlap area
% created on 8/16/2025 by Xuemei Chen

%[num_rows, num_cols] = data_size;
num_rows = data_size(1);
num_cols = data_size(2);

[Y_nrows, Y_ncols] = size(Y);

%{
overlap_length = floor(Y_nrows*overlap);
nonlap_length = Y_nrows - overlap_length;
temp = floor((num_rows - Y_nrows)/nonlap_length);
if temp < (num_rows - Y_nrows)/nonlap_length
    multiplier = temp + 2;
else
    multiplier = temp + 1;
end
%}

multiplier = ceil(Y_ncols/num_cols);

idx = chop_data((1:num_rows)', num_of_chops, overlap);

data = zeros(data_size);

%DIFF = idx(1,2) - idx(1,1);
idx_diff = idx(1,:) - idx(1, 1);

for ci = 1:num_cols
    j = 1;
    
    nonoverlap_idx = setdiff(idx(:,j), idx(:,j+1));
    data(nonoverlap_idx, ci) = Y(nonoverlap_idx,j+(ci-1)*multiplier);

    overlap_idx = intersect(idx(:,j), idx(:,j+1));
    data(overlap_idx, ci) = 0.5*(Y(overlap_idx-idx_diff(j),j+(ci-1)*multiplier) + Y(overlap_idx-idx_diff(j+1),j+1+(ci-1)*multiplier));

    for j = 2:(multiplier-1)
        nonoverlap_idx = setdiff(idx(:,j), idx(:,j+1));
        % discard if already covered by previous j
        nonoverlap_idx2 = setdiff(nonoverlap_idx, idx(:,j-1));
        data(nonoverlap_idx2, ci) = Y(nonoverlap_idx2-idx_diff(j),j+(ci-1)*multiplier);

        overlap_idx = intersect(idx(:,j), idx(:,j+1));
        %keyboard
        data(overlap_idx, ci) = 0.5*(Y(overlap_idx-idx_diff(j),j+(ci-1)*multiplier) + Y(overlap_idx-idx_diff(j+1),j+1+(ci-1)*multiplier));
    
    end
    j = multiplier;
    need_idx = setdiff(idx(:,j), idx(:,j-1));

    
    data(need_idx, ci) = Y(need_idx-idx_diff(j),j+(ci-1)*multiplier);
    
end
end

