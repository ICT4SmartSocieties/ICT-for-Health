%%%%%%%%%
% Lab02 %
%%%%%%%%%

clear all
close all 
clc

newMatrix = loadData();
[data_train_norm, data_test_norm] = cleanData(newMatrix);

[X_train, y_train, X_test, y_test, a, a_L]= PCE(data_train_norm, data_test_norm, 7);

yhat_train = X_train*a;
yhat_train_L = X_train * a_L;
yhat_test = X_test*a;
yhat_test_L = X_test*a_L;

err1 = immse(y_train-yhat_train, y_train-yhat_train_L);
err2 = immse(y_test - yhat_test, y_test-yhat_test_L);

%%%%%%%%%%%%%%%%%%%%
% plots generation %
%%%%%%%%%%%%%%%%%%%%

%%1
figure
plot(yhat_train,y_train, 'o');
title('yhat\_train versus y\_train')

figure
plot(yhat_test_L,y_test, 'o');
title('yhat\_test\_L versus y\_test')

%%2
figure
plot(yhat_test, y_test, 'o')
title('yhat\_test versus y\_test')

figure 
plot(yhat_test_L, y_test,'o')
title('yhat\_test\_L versus y\_test')

%%3
figure
hist(y_train - yhat_train, 50)
title('y\_train - yhat\_train')
hold on
figure
hist(y_train-yhat_train_L,50)
title('y\_train - yhat\_train\_L')

%4
figure
hist(y_test - yhat_test, 50)
title('y\_test - yhat\_test')
hold on
figure
hist(y_test-yhat_test_L,50)
title('y\_test - yhat\_test\_L')

%%5
figure
plot(a,'o')
hold on
plot(a_L,'o')
legend('a','a_L')







