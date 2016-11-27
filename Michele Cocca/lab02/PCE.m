function [X_train, y_train, X_test, y_test, a, a_L]=PCE (data_train_norm, data_test_norm, F0)
    y_train=data_train_norm(:,F0);
    X_train=data_train_norm(:,5:22);
    X_train(:,F0)=[];

    y_test=data_test_norm(:,F0); 
    X_test=data_test_norm(:,5:22);
    X_test(:,F0)=[];

    a = pinv(X_train)*y_train;
    y_hat_test = X_test*a;

    %evalutation of a_L
    [X_train_R, X_train_C] = size(X_train);
    R = (1/X_train_R)*X_train'* X_train;
    [V,D] = eig(R); % v-> eigenVECT D->eigenVAL in diagonal matrix;
    lambdas = diag(D);

    %take the 90%of eigenVAL
    P = sum(lambdas);
    i=0;
    partialSum=0;
    while partialSum < 0.9
        i=i+1;
        partialSum = partialSum +(lambdas(i)/P);
    end
    LAMBDA_L  = D(1:i,1:i);
    U_L = V(:,1:i);
    a_L = (1/X_train_R)*U_L* inv(LAMBDA_L) * U_L' * X_train'*y_train;
end