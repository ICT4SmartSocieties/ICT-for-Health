function a=MSEprediction(data_train_norm, data_test_norm, F0)
    y_train = data_train_norm(:,F0);
    X_train = data_train_norm(:,5:22);
    X_train(:,F0)=[];

    y_test=data_test_norm(:,F0);
    X_test=data_test_norm(:,5:22);
    X_test(:,F0)=[];
    a=[];
    a = pinv(X_train) * y_train;
    
    yhat_train=X_train*a;
    figure
    plot(yhat_train,'.')
    hold on
    plot(y_train, '--')
    title(['MSE. F0=',num2str(F0)]);
    legend('yhat\_train','y\_train')
    figure
    histogram(yhat_train-y_train,50)
    title(['MSE. Difference yhat - ytrain. F0=',num2str(F0)]);
end