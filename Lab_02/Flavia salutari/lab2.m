close all
clear all
clc

load('updrs.mat')
totalpatients = parkinsonsupdrs(size(parkinsonsupdrs,1),1);
matricepazienti = zeros(1,22);
for k = 1:totalpatients
    patient_matrix = parkinsonsupdrs(find(parkinsonsupdrs(:,1)==k),:);
    patient_matrix(:,4) = abs(fix(patient_matrix(:,4)));
    days_patient = unique(patient_matrix(:,4));
    for i = 1:length(days_patient)
        day = days_patient(i);
        indexes_day = find(patient_matrix(:,4)==day);
        new_matrix = mean(patient_matrix(indexes_day,:),1);
        matricepazienti = [matricepazienti;new_matrix];
    end
end 

matricepazienti = matricepazienti(2:end,:);


%%% Perform regression [1] 
data_train=zeros(1,22);
data_test=zeros(1,22);
for j = 1 : 36
    data_train_mid = matricepazienti(find(matricepazienti(:,1)==j),:);
    data_train = [data_train;data_train_mid];
end
data_train = data_train(2:end,:);

for j = 37 : 42
    data_test_mid = matricepazienti(find(matricepazienti(:,1)==j),:);
    data_test = [data_test;data_test_mid];
end
data_test = data_test(2:end,:);

m_data_train=mean(data_train,1);
v_data_train=var(data_train,1);

for i = 1:size(data_train,1)
    data_train_norm(i, 1:4) = data_train(i, 1:4);
    for z = 5:22
        data_train_norm(i, z) = (data_train(i, z) - m_data_train(z)) / sqrt(v_data_train(z));
    end
end
mean(data_train_norm,1); % verify it is zero mean and va = 1?? 
var(data_train_norm,1);

for i = 1:size(data_test,1)
    data_test_norm(i, 1:4) = data_test(i, 1:4);
    for z = 5:22
        data_test_norm(i, z) = (data_test(i, z) - m_data_train(z)) / sqrt(v_data_train(z));
    end
end
mean(data_train_norm,1); % verify it is zero mean and va = 1?? 
var(data_train_norm,1);

% Perform regression [2]
F0 = 7; % // 5
y_train=data_train_norm(:,F0); %% feature che elimino e poi vorrÚ stimare
X_train=data_train_norm;
X_train(:,F0)=[];  %% tutte le feature senza F0 

y_test=data_test_norm(:,F0); 
X_test=data_test_norm;
X_test(:,F0)=[];

%%% PCR
N = size(X_train,1);
R = (1/N) * X_train(:,5:end).' * X_train(:,5:end);
[U, A] = eig(R);
Z = X_train(:, 5:end) * U;

Z_norm = (1 / sqrt(N)) * Z * A^(-1/2);
Z_y = Z_norm.' * y_train;
y_hat = Z_norm * Z_y;
% a = inv(X_train*X_train.') * X_train * y_train;
a = (1/N) * U * inv(A) * U.' * X_train(:,5:end).' * y_train;
y_hat_2_train = X_train(:,5:end) * a; %%% ?????????????????????

figure
plot(y_hat_2_train, '-y')
hold on
plot(y_train)
grid on
title('y train')

% test

y_hat_test = X_test(:,5:end) * a;

figure
plot(y_hat_test, '-y')
hold on
plot(y_test)
grid on
title('y test')


%%%%%%%%%%%% L %%%%%%%%%%
K = 1;
L = 5;
U_L = U(:,K:L);
A_L = A(K:L,K:L);

Z_norm_L = 1/sqrt(N) * X_train(:,5:end) * U_L * A_L ^(-1/2);

Z_y_L = Z_norm_L.' * y_train;

y_hat_L = Z_norm_L * Z_y_L;

a_hat_L = 1/N * U_L * inv(A_L) * U_L.' * X_train(:,5:end).' *y_hat_L;

stima_L = X_train(:,5:end) * a_hat_L;
stima_L_2 = X_test(:,5:end) * a_hat_L;

errore = norm(stima_L_2-y_test);

figure
plot(stima_L)
hold on
plot(y_train)
grid on
title('With L, train')

figure
plot(stima_L_2)
hold on
plot(y_test)
grid on
title('With L, test')