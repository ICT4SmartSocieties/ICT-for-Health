close all
clear all
%Lab 1 - ICT HEALTH - Perform regression - Steepest Descent

load('data_train_norm.mat');
load('data_test_norm.mat');

F0 = 7;

y_train = data_train_norm(:,F0);
X_train = data_train_norm(:,5:22);
X_train(:,F0) = [];

y_test =data_test_norm(:,F0);
X_test=data_test_norm(:,5:22);
X_test(:,F0)=[];

rng('default');

a_hat_i = rand(17,1);
a_hat_ii = rand(17,1);
epsilon = 10^-6;

counter=0;
while ( norm(a_hat_i - a_hat_ii) > epsilon )
    
    grad_a_hat_i = -2 * transpose(X_train)*y_train + 2 * transpose(X_train)*X_train*a_hat_i;
    H = 4 * transpose(X_train) * X_train;
    a_hat_ii = a_hat_i;
    a_hat_i = a_hat_i - norm(grad_a_hat_i)^2*grad_a_hat_i/( transpose(grad_a_hat_i)*H*grad_a_hat_i );
    counter = counter + 1;
    
end

y_hat_train = X_train * a_hat_i;
y_hat_test = X_test * a_hat_i;

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