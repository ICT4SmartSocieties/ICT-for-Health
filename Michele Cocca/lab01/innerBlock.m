function s = innerBlock(pID, line, time, groupedMatrix)
    [r,c] = size(groupedMatrix);
    flag=0;
    s=0;
    while flag == 0
        if groupedMatrix(line,1) == pID && groupedMatrix(line, 4) == time && line < r
            s=s+1;
            line= line+1;
        else
            flag = 1;
        end
    end
end