function output = normalize_matrix(mat)
    mean_mat = mean(mat);
    std_mat = std(mat);
    output = (mat-mean_mat)./std_mat;
end