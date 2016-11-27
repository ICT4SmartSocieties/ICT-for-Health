function [data_train_norm, data_test_norm]=cleanData(newMatrix)
    %select measurment of first 36 patients
    data_train=[];
    line=1;
    p = newMatrix(line,1);
    while p<=36
        p = newMatrix(line,1);
        data_train = [data_train; newMatrix(line,:)];
        line=line + 1;
    end 

    data_test=[];
    while line <= 955 
       p = newMatrix(line,1);
       data_test = [data_test; newMatrix(line,:)];
       line = line+1;
    end


    %normalizzation
    m_data_train=mean(data_train(:,5:22),1); %mean for each column
    v_data_train=var(data_train(:,5:22),1); %variance normalized for #elements -1

    %normalizing on the column 7-22 by using the  m/v vectors
    data_train_norm = data_train(:,1:4);
    [r,c] = size(data_train);
    for line=1:r
        for col=5:c
            data_train_norm(line,col)=(data_train(line,col)-m_data_train(col-4)) / sqrt(v_data_train(col-4));
        end
    end

    [r,c] = size(data_test);
    data_test_norm=data_test(:,1:4);
    for line=1:r
        for col=5:c
            data_test_norm(line,col) = (data_train(line,col)-m_data_train(col-4))/sqrt(v_data_train(col-4));
        end
    end
end 