function a=gradientAlgorithm(data_train_norm, data_test_norm, F0)
    y_train = data_train_norm(:,F0);
    X_train = data_train_norm(:,5:22);
    X_train(:,F0)=[];

    y_test=data_test_norm(:,F0);
    X_test=data_test_norm(:,5:22);
    X_test(:,F0)=[];

    rng('default');
    a = rand(17,1);
    a_next = a*2;
    epsilon = 1e-5;
    gamma = 3e-8;
    i=1;

    while norm(a_next - a) > epsilon
        a = a_next;
        gradErr = -2*X_train' *y_train + 2*X_train'*X_train*a;
        a_next = a - gamma*gradErr; 
        i=i+1;
    end
    yhat_train=X_train*a;
    figure
    plot(yhat_train,'.')
    hold on
    plot(y_train, '--')
    title(['GA. F0=',num2str(F0)]);
    legend('yhat\_train','y\_train')
    figure
    histogram(yhat_train-y_train,50)
    title(['GA. Difference yhat - ytrain. F0=',num2str(F0)]);
end