close all
clear all
%Lab 1 - ICT HEALTH - Perform regression - Gradient Algorithm

load('data_train_norm.mat');
load('data_test_norm.mat');

F0 = 7;

y_train = data_train_norm(:,F0);
X_train = data_train_norm;
X_train(:,F0) = [];

y_test=data_test_norm(:,F0);
X_test=data_test_norm;
X_test(:,F0)=[];

rng('default');

a_hat = rand(21,1);
gamma = 0.2;
epsilon = 10^-6;
a_hat_old = rand(21,1);

% while ( norm(a_hat - a_hat_old) > epsilon )
%     grad_a_hat = -2 * transpose(X_train)*y_train + 2 * transpose(X_train)*X_train*a_hat;
%     tmp_a_hat = a_hat - (gamma * grad_a_hat);
%     a_hat_old = a_hat;
%     a_hat = tmp_a_hat;
% end

for i = 1:10000

    gradiente = gradient (a_hat);
    a_hat = a_hat - gamma*gradiente;
    
end




% counter=0;
% for i=1:10000
%     grad_a_hat = gradient(a_hat);
%     a_hat_old = a_hat;
%     a_hat = a_hat - gamma*grad_a_hat;
% %     grad_a_hat = -2 * transpose(X_train)*y_train + 2 * transpose(X_train)*X_train*a_hat;
% %     grad_a_hat = gradient(a_hat);
% %     a_hat = a_hat - (gamma * grad_a_hat);
%     counter = counter+1;
% end