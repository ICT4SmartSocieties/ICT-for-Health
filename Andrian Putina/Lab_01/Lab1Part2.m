close all
clear all
%Lab 1 - ICT HEALTH - Prepare data for train

load('finalMatrix.mat');

Npacients = 36;
rowPacients_train = finalMatrix(:,1) <= Npacients;
rowPacients_test =  finalMatrix(:,1) > Npacients;

data_train = finalMatrix(rowPacients_train,:);
    m_data_train = mean(data_train,1);
    v_data_train = var(data_train,1);
    std_data_train = std(data_train,1);

    data_train_norm = data_train(:,5:22) - m_data_train(:,5:22);
    data_train_norm = data_train_norm ./ std_data_train(:,5:22);
    data_train_norm = [data_train(:,1:4), data_train_norm];
    
data_test = finalMatrix(rowPacients_test,:);   
    data_test_norm = data_test(:,5:22) - m_data_train(:,5:22);
    data_test_norm = data_test_norm ./ std_data_train(:,5:22);
    data_test_norm = [data_test(:,1:4), data_test_norm];

save('data_train_norm','data_train_norm');
save('data_test_norm','data_test_norm');