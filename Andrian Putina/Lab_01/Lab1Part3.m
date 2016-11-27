close all
clear all
%Lab 1 - ICT HEALTH - Perform regression - MSE

load('data_train_norm.mat');
load('data_test_norm.mat');

F0 = 7;

y_train = data_train_norm(:,F0);
X_train = data_train_norm;
X_train(:,F0) = [];

y_test=data_test_norm(:,F0);
X_test=data_test_norm;
X_test(:,F0)=[];

% Estimate a_hat
a_hat = inv(transpose(X_train)*X_train)*transpose(X_train)*y_train;
y_hat_train = X_train * a_hat;
y_hat_test = X_test * a_hat;

figure
plot(y_hat_train)
hold on
plot(y_train, '--k')
axis([0 840 -2 18])
grid on
legend('y\_hat\_train','y\_train', 'Location', 'northwest')
title('y\_hat\_train vs y\_train')

figure
plot(y_test)
hold on
plot(y_hat_test, '--k')
axis([0 150 -1.75 3.5])
grid on
legend('y\_test', 'y\_hat\_test')
title('y\_test vs y\_hat\_test')

[N,X] = hist(y_hat_train - y_train, 10);
figure
hist(y_hat_train - y_train, 10)
grid on


[N2,X2] = hist(y_hat_test - y_test, 10);
figure
hist(y_hat_test - y_test, 10)
grid on

% a_hat2 = inv(transpose(X_train(:,5:end))*X_train(:,5:end))*transpose(X_train(:,5:end))*y_train;
% y_hat_train2 = X_train(:,5:end) * a_hat2;
% y_hat_test2 = X_test(:,5:end) * a_hat2;
% 
% figure
% plot(y_hat_train2)
% hold on
% plot(y_train)
% 
% figure
% plot(y_hat_test2)
% hold on
% plot(y_test)
% 
% 
% figure
% hist(y_hat_train2 - y_train,200)
% 
% figure
% hist(y_hat_test2 - y_test,200)