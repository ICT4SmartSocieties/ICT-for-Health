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
mean(data_train_norm,1) % verify it is zero mean and va = 1?? 
var(data_train_norm,1)

for i = 1:size(data_test,1)
    data_test_norm(i, 1:4) = data_test(i, 1:4);
    for z = 5:22
        data_test_norm(i, z) = (data_test(i, z) - m_data_train(z)) / sqrt(v_data_train(z));
    end
end
mean(data_train_norm,1) % verify it is zero mean and va = 1?? 
var(data_train_norm,1)

% Perform regression [2]
F0 = 5;
y_train=data_train_norm(:,F0); %% feature che elimino e poi vorrò stimare
X_train=data_train_norm;
X_train(:,F0)=[];  %% tutte le feature senza F0 

y_test=data_test_norm(:,F0); 
X_test=data_test_norm;
X_test(:,F0)=[];

%% MSE
a_hat = inv(X_train(:,5:end).' * X_train(:,5:end)) * X_train(:,5:end).' * y_train; % 

% stima valori _ train
y_train_hat = X_train(:,5:end) * a_hat;

figure
plot(y_train_hat)
hold on
plot(y_train, '-r')

% stima valori _ test
y_test_hat = X_test(:,5:end) * a_hat;

figure
plot(y_test_hat)
hold on
plot(y_test, '-r')

%%% istogrammi
figure
hist(y_train - y_train_hat, 50)
figure
hist(y_test - y_test_hat, 50) %% dagli istogrammi capiamo che la dist
% degli errori è circa una gaussiana? a cosa è dovuto?

%%% CHIEDERE !!!!!! a^ che abbiamo trovato, per trovare la 7esima feature
%%% si fa la combinazione lineare di a_hat * la riga relativa a un paziente
%%% con tutte le sue features?

%% the gradient algorithm
rng('default')
a_i = rand(21,1);
gamma = 10^-3;
epsilon = 10^-4;
a2 = zeros(21,1);

while (norm(a_i - a2) > epsilon)
    grad_a_i = - 2* X_train.' * y_train + 2 * X_train.' * X_train * a_i;
    a_ii = a_i - (gamma * grad_a_i);
    a2 = a_i;
    a_i = a_ii;
end



%% steepest descent 
rng('default')
a_i = rand(21,1);
epsilon = 10^-4;
a2 = zeros(21,1);
i= 1;
while (norm(a_i - a2) > epsilon)
    grad_a_i = - 2* X_train.' * y_train + 2 * X_train.' * X_train * a_i;
    hess_a_i = 4 * X_train.' * X_train;
    a_ii = a_i - ((norm(grad_a_i)^2 * grad_a_i)/(grad_a_i.' * hess_a_i * grad_a_i));
    a2 = a_i;
    a_i = a_ii;
end

