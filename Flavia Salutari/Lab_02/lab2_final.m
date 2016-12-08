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
matricepazienti_norm = (matricepazienti - (ones(size(matricepazienti,1),1)*mean(matricepazienti))) ./ sqrt(ones(size(matricepazienti,1),1)*var(matricepazienti));

%%% Perform regression [1] 

data_train = matricepazienti(matricepazienti(:,1)<37,:);
data_test = matricepazienti(matricepazienti(:,1)>36,:);

m_data_train=mean(data_train);
v_data_train=var(data_train);
o = ones(size(data_train,1),1);
data_train_norm = data_train;
data_train_norm(:,5:end) = (data_train(:,5:end) - o*m_data_train(:,5:end)) ./ sqrt(o*v_data_train(:,5:end));

mean_train_norm = mean(data_train_norm,1);  
var_train_norm = var(data_train_norm,1); 

o = ones(size(data_test,1),1);
data_test_norm = data_test;
data_test_norm(:,5:end) = (data_test(:,5:end) - o*m_data_train(:,5:end)) ./ sqrt(o*v_data_train(:,5:end));

mean_test_norm = mean(data_train_norm,1);
var_test_norm = var(data_train_norm,1);

% Perform regression 
F0 = 7;
y_train=data_train_norm(:,F0); %% feature che elimino e poi vorr? stimare
X_train=data_train_norm;
X_train(:,F0)=[];  %% tutte le feature senza F0 

y_test=data_test_norm(:,F0); 
X_test=data_test_norm;
X_test(:,F0)=[];
%%
%%% PCR
N = size(X_train,1);
R = (1/N) * X_train(:,5:end).' * X_train(:,5:end);
[U, A] = eig(R);
% 
% 
% Z = X_train(:, 5:end) * U;
% Z_norm = (1 / sqrt(N)) * Z * A^(-1/2);
% Z_y = Z_norm.' * y_train;
% y_hat = Z_norm * Z_y;

a = (1/N) * U * inv(A) * U.' * X_train(:,5:end).' * y_train;
y_hat_train = X_train(:,5:end) * a; %%% ?????????????????????

MSE_train = mean((y_hat_train-y_train).^2)

figure
plot(y_hat_train, '-k')
hold on
plot(y_train)
grid on
title('$\hat{y}\_train vs y\_train$','Interpreter','latex')
legend('$\hat{y}\_train$','$y\_train$')
set(legend,'Interpreter','latex')
% test

y_hat_test = X_test(:,5:end) * a;
MSE_test = mean((y_hat_test-y_test).^2)

figure
plot(y_hat_test, '-k')
hold on
plot(y_test)
grid on
title('$\hat{y}\_test vs y\_test$','Interpreter','latex')
legend('$\hat{y}\_test$','$y\_test$')
set(legend,'Interpreter','latex')

%%% istogrammi
figure
hist(y_train - y_hat_train, 50)
title('$\hat{y}\_train - y\_train$','Interpreter','latex')
figure
[nb,xb] =hist(y_test - y_hat_test, 50, 'r');
bh=bar(xb,nb);
set(bh,'facecolor',[1 0 0]);
title('$\hat{y}\_test - y\_test$','Interpreter','latex')

%%%%%%%%%%%% L %%%%%%%%%%

K = 1;
total_eig = sum(diag(A));
perc = 0.999;
percentage_thresh = perc * total_eig;

d = diag(A);
dcum = cumsum(d);
L = length(find(dcum<percentage_thresh));

U_L = U(:,K:L);
A_L = A(K:L,K:L);

% Z_norm_L = 1/sqrt(N) * X_train(:,5:end) * U_L * A_L ^(-1/2);
% Z_y_L = Z_norm_L.' * y_train;
% y_hat_L = Z_norm_L * Z_y_L;

a_hat_L = 1/N * U_L * inv(A_L) * U_L.' * X_train(:,5:end).' *y_train;

y_hat_train_L = X_train(:, 5:end) * a_hat_L;
y_hat_test_L = X_test(:, 5:end) * a_hat_L;

MSE_L_train = mean((y_hat_train_L-y_train).^2)
MSE_L_test = mean((y_hat_test_L-y_test).^2)

figure
plot(y_hat_train_L)
hold on
plot(y_train)
grid on
str = sprintf('With L = %d features - train, F0 = %d, ThresholdPercentage = %d', L, F0, perc);
title(str)
legend('$\hat{y}\_train\_L$','$y\_train$')
set(legend,'Interpreter','latex')

figure
plot(y_hat_test_L)
hold on
plot(y_test)
grid on
str = sprintf('With L = %d features - test, F0 = %d, ThresholdPercentage = %d', L, F0, perc);
title(str)
legend('$\hat{y}\_test\_L$','$y\_test$')
set(legend,'Interpreter','latex')


%%% istogrammi
figure
hist(y_train - y_hat_train_L, 50)
title('$\hat{y}\_train\_L - y\_train$','Interpreter','latex')
figure
[nb,xb] =hist(y_test - y_hat_test_L, 50, 'r');
bh=bar(xb,nb);
set(bh,'facecolor',[1 0 0]);
title('$\hat{y}\_test\_L - y\_test$','Interpreter','latex')


figure
plot(a)
hold on
plot(a_hat_L, 'k')
grid on
legend('$\hat{a}$', '$\hat{a}\_L$')
set(legend,'Interpreter','latex')
