close all
clear all
% Lab 2 - ICT HEALTH - PCR

load('data_train_norm.mat');
load('data_test_norm.mat');

F0 = 7;
N= 840;

%Prepare data 
y_train = data_train_norm(:,F0);
X_train = data_train_norm(:,5:22);
X_train(:,F0) = [];

y_test = data_test_norm(:,F0);
X_test = data_test_norm(:,5:22);
X_test(:,F0)=[];

%Build R matrix
%We can say that R(i,k) ~ 0 if feature i and k are uncorrelated
R = 1/N*transpose(X_train)*X_train;
[P,D] = eig(R);
Diag = inv(P)*R*P;
Diag2 = P*D*transpose(P);
Z=X_train*P;

%Rnuovo = 1/N*transpose(Z)*Z;

Z_norm = 1/sqrt(N) * Z * D ^ (-1/2);
Zy = transpose(Z_norm)*y_train
y_hat = 1/N*Z_norm*Zy;

a = P*inv(D)*transpose(P)*transpose(X_train)*y_hat;

stima = X_train * a;
stima2 = X_test * a;

figure
plot(stima)
hold on
plot(y_train)
grid on
title('Without L, y train')

figure
plot(stima2)
hold on
plot(y_test)
grid on
title('Without L, y test')

%Reduce the number of features L
K = 1;
L = 5;
D_L = D(K:L,K:L);
P_L = P(:,K:L);

Z_norm_L = 1/sqrt(N) * X_train * P_L * D_L ^ (-1/2);

Z_y_L = transpose(Z_norm_L)*y_train;

y_hat_L = Z_norm_L * Z_y_L;

a_hat_L = 1/N * P_L * inv(D_L) * transpose(P_L) * transpose(X_train) *y_hat_L;

stima_L = X_train * a_hat_L;
stima_L_2 = X_test * a_hat_L;

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