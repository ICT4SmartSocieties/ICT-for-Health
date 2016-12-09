close all
clear all
clc

load('updrs.mat');
updrs = parkinsonsupdrs;
nOfDays = 180;
nOfPatients = 42;
lungh = 0;
count = 1;
set(groot,'DefaultLegendInterpreter','latex')
set(groot,'DefaultTextInterpreter','latex')

% ========================= DATA LOADING =============================
% ====================================================================
for patient = 1:nOfPatients
    patientIndex = [];  
    timeArray = [];
    patientIndex = find(updrs(:, 1) == patient);    % Index-Patient vector
    
    for k = 1:length(patientIndex)
        % timeArray --> is the vector containing all the days rounded to
        % the closest INTEGER number related to the "patient"-th patient.
        timeArray(k, 1) = floor(updrs(patientIndex(k), 4)); 
    end
    
    % In this cycle I evaluate the mean for each patient: I scroll every
    % day from day 1 to day 180: in this way I do not consider negative
    % days.
    for days = 1:nOfDays
        timeIndex = [];
        sumRow = zeros(1, 22); 
        timeIndex = find(timeArray == days);    % Contains the indices of 
                                                % the days index
        
        if ~isempty(timeIndex)  
            for (ii = 1:length(timeIndex))
                sumRow = sumRow + updrs(timeIndex(ii) + lungh, :);
            end
            sumRow = sumRow ./ length(timeIndex);   % <-- Contains the 
                                                    % vector of the means
            updrsNew(count, :) = sumRow;    % <-- Final matrix for the 
                                            % patient "patient"-th
            count = count + 1;
        end
    end 
    % lungh contains the index of the last element related to the
    % "patient"-th patiens.
    lungh = patientIndex(length(patientIndex)); 
end

train = 1:36;
test = 37:nOfPatients;
trainIndex = find(updrsNew(:, 1) < 37);
testIndex = find(updrsNew(:, 1) > 36);
data_train = updrsNew(1:length(trainIndex), :);
data_test = updrsNew(length(trainIndex)+1:end, :);
m_data_train = mean(data_train, 1);
v_data_train = var(data_train, 1);

% -------------------- Data Normalization ----------------------
% In the normalization I just consider the columns related to the features:
% from 5 to 22.
trainDim = length(data_train(:, 1));
testDim = length(data_test(:, 1));
onesMatrixTrain = ones(trainDim, 1);
onesMatrixTest = ones(testDim, 1);
meanMatrixTrain = onesMatrixTrain * m_data_train(5:22);
meanMatrixTest = onesMatrixTest * m_data_train(5:22);
varMatrixTrain = onesMatrixTrain * v_data_train(5:22);
varMatrixTest = onesMatrixTest * v_data_train(5:22);
onesMatrixTest = ones(testDim, 1);
media = m_data_train(5:22); 
data_train_noMeanVar = (data_train(:, 5:22) - meanMatrixTrain) ./ ...
                       sqrt(varMatrixTrain);
data_test_noMeanVar = (data_test(:, 5:22) - meanMatrixTest) ./ ...
                       sqrt(varMatrixTest);

% NORMALIZED DATASET for both training and testing.
data_train_norm = data_train_noMeanVar;
data_test_norm = data_test_noMeanVar;

DataNorm = [data_train_norm; data_test_norm];

F0 = [1 3]; % Vector containing analysed features.

% ========================= MSE ESTIMATION ===========================
% ====================================================================
for ii = 1:length(F0)
    y_train = data_train_norm(:, F0(ii));   
    x_train = data_train_norm;
    x_train(:, F0(ii)) = [];    

    y_test = data_test_norm(:, F0(ii));
    x_test = data_test_norm;
    x_test(:, F0(ii)) = [];

    aHat = pinv(x_train) * y_train;     % pinv function gives back the pse-
                                        % udo-inverse of the x matrix.
    y_train_hat = x_train * aHat;       % Trainin result. 
    y_test_hat = x_test * aHat;         % Testing result.
    
    % Task 1
    figure, subplot(2,1,1)
    plot(y_train_hat, '--*'), hold on, grid on, plot(y_train, '')
    title(['MSE regression: TRAIN F0 = ', num2str(F0(ii))]) 
    legend('$\hat{y}$\_train', 'y\_train')
    
    % Task 2
    subplot(2,1,2)
    plot(y_test_hat, '--*'), hold on, grid on, plot(y_test, '')
    title(['MSE regression: TEST F0 = ', num2str(F0(ii))])
    legend('$\hat{y}$\_test', 'y\_test')
    
    figure, subplot(2,1,1)
    plot(y_train, y_train_hat, 'o'), grid on
    title(['MSE: Regression for F0 = ', num2str(F0(ii))])
    xlabel('y\_train'), ylabel('$\hat{y}$\_train')
    
    % Tast 5
    subplot(2,1,2), plot(aHat), grid on
    title(['MSE: Coefficients w for F0 = ', num2str(F0(ii))])    
    
    % Task 3
    figure, subplot(2,1,1)
    hist(y_train - y_train_hat, 50), grid on
    title(['MSE: Prediction error on TRAIN for F0 = ', num2str(F0(ii))])    

    % Task 4
    subplot(2,1,2)
    hist(y_test - y_test_hat, 50), grid on
    title(['MSE: Prediction error on TEST for F0 = ', num2str(F0(ii))])
end

% ========================= GRADIENT ALGORITHM =======================
% ====================================================================
rng('default');
M = length(aHat);   % Number of features
epsilon = 10^-6;    % Treshold for the stopping condition
gamma = 10^-4;      % Speed of convergence
F0 = [1 3];
countGA = zeros(1, length(F0));     % Vector containing the number of 
                                    % iterations per each F0

for ii = 1:length(F0)
    y_train = data_train_norm(:, F0(ii));
    x_train = data_train_norm;
    x_train(:, F0(ii)) = [];    

    y_test = data_test_norm(:, F0(ii));
    x_test = data_test_norm;
    x_test(:, F0(ii)) = [];
    
    % Coefficients vector - gradient - gamma - coefficients vector 
    % updated INITIALIZATION
    aHatInitialGA = rand(M, 1);
    gradientGA = (-2 * (x_train)' * y_train) + (2 * (x_train)' * ...
                  x_train * aHatInitialGA);
    aHatFinalGA = aHatInitialGA - (gamma * gradientGA);
    
    % aHatInitial = a(i)
    % aHatFinal = a(i + 1) --> VETTORE FINALE DI COEFFICIENTI
    while norm(aHatFinalGA - aHatInitialGA) > epsilon
        countGA(ii) = countGA(ii) + 1;
        aHatInitialGA = aHatFinalGA;
        gradientGA = (-2 * x_train' * y_train) + (2 * x_train' * ...
                      x_train * aHatInitialGA);
        aHatFinalGA = aHatInitialGA - (gamma * gradientGA);
    end
    % In aHatFinal there is the final set of coefficients a(i+1):
    y_train_hat = x_train * aHatFinalGA;
    y_test_hat = x_test * aHatFinalGA;
    
    % Task 1
    figure, subplot(2,1,1)
    plot(y_train_hat, '--*'), hold on, grid on, plot(y_train, '')
    title(['GRADIENT ALGORITHM: TRAIN F0 = ', num2str(F0(ii))])
    legend('$\hat{y}$\_train', 'y\_train')

    % Task 2
    subplot(2,1,2)
    plot(y_test_hat, '--*'), hold on, grid on, plot(y_test, '')
    title(['GRADIENT ALGORITHM: TEST F0 = ', num2str(F0(ii))])
    legend('$\hat{y}$\_test', 'y\_test')
    
    figure, subplot(2,1,1)
    plot(y_train, y_train_hat, 'o'), grid on
    title(['GRADIENT ALGORITHM: Prediction trend for F0 = ', ...
          num2str(F0(ii))])
    xlabel('y\_train'), ylabel('$\hat{y}$\_train')
    
    % Tast 5
    subplot(2,1,2), plot(aHatFinalGA), grid on
    title(['GRADIENT ALGORITHM: Coefficients w for F0 = ', ...
        num2str(F0(ii))])        
    
    % Task 3
    figure, subplot(2,1,1)
    hist(y_train - y_train_hat, 50), grid on
    title(['GRADIENT ALGORITHM: Prediction error on TRAIN for F0 = ', ...
        num2str(F0(ii))])    

    % Task 4
    subplot(2,1,2)
    hist(y_test - y_test_hat, 50), grid on
    title(['GRADIENT ALGORITHM: Prediction error on TEST for F0 = ', ...
        num2str(F0(ii))])    
end

% ========================= STEPEST DESCENT ==========================
% ====================================================================
rng('Default');
threshold = 10^-6;
F0 = [1 3];
countSD = zeros(1, length(F0));     % Vector containing the number of 
                                    % iterations per each F0

for ii = 1:length(F0)
    y_train = data_train_norm(:, F0(ii));
    x_train = data_train_norm;
    x_train(:, F0(ii)) = [];    

    y_test = data_test_norm(:, F0(ii));
    x_test = data_test_norm;
    x_test(:, F0(ii)) = [];
    
    % Coefficients vector - gradient - gamma - coefficients vector 
    % updated INITIALIZATION
    aHatInitialSD = rand(M, 1);
    gradientSD = (-2 * (x_train)' * y_train) + (2 * (x_train)' * ...
                  x_train * aHatInitialSD);
    hessianAHat = 4 * (x_train') * x_train;
    gammaSD = ((norm(gradientSD)^2) / (gradientSD' * hessianAHat * ...
               gradientSD));
    aHatFinalSD = aHatInitialSD - (gammaSD * gradientSD);
    
    while norm(aHatFinalSD - aHatInitialSD) > threshold
        countSD(ii) = countSD(ii) + 1;
        aHatInitialSD = aHatFinalSD;
        gradientSD = (-2 * (x_train') * y_train) + (2 * (x_train)' *  ...
                      x_train * aHatInitialSD);
        gammaSD = ((norm(gradientSD)^2) / (gradientSD' * hessianAHat * ...
                   gradientSD));
        aHatFinalSD = aHatInitialSD - (gammaSD * gradientSD);
    end
    
    y_train_hat = x_train * aHatFinalSD;
    y_test_hat = x_test * aHatFinalSD;    
    
    % Task 1
    figure, subplot(2,1,1)
    plot(y_train_hat, '--*'), hold on, grid on, plot(y_train, '')
    title(['STEEPEST DESCENT: TRAIN F0 = ', num2str(F0(ii))])
    legend('$\hat{y}$\_train', 'y\_train')

    % Task 2
    subplot(2,1,2)
    plot(y_test_hat, '--*'), hold on, grid on, plot(y_test, '')
    title(['STEEPEST DESCENT: TEST F0 = ', num2str(F0(ii))])
    legend('$\hat{y}$\_test', 'y\_test')

    figure, subplot(2,1,1)
    plot(y_train, y_train_hat, 'o'), grid on
    title(['STEEPEST DESCENT: Prediction trend for F0 = ', ...
        num2str(F0(ii))])
    xlabel('y\_train'), ylabel('$\hat{y}$\_train')  
    
    % Tast 5
    subplot(2,1,2), plot(aHatFinalSD), grid on
    title(['STEEPEST DESCENT: Coefficients w for F0 = ', num2str(F0(ii))])    
        
    % Task 3
    figure, subplot(2,1,1)
    hist(y_train - y_train_hat, 50), grid on
    title(['STEEPEST DESCENT: Prediction error on TRAIN for F0 = ', ...
        num2str(F0(ii))])    

    % Task 4
    subplot(2,1,2), hist(y_test - y_test_hat, 50), grid on
    title(['STEEPEST DESCENT: Prediction error on TEST for F0 = ', ...
        num2str(F0(ii))])      
end

% =============================== PCR ================================
% ====================================================================
F0 = [1 3];

for ii = 1:length(F0)
    y_train = data_train_norm(:, F0(ii));
    x_train = data_train_norm;
    x_train(:, F0(ii)) = [];    

    y_test = data_test_norm(:, F0(ii));
    x_test = data_test_norm;
    x_test(:, F0(ii)) = [];
    
    N = length(y_train);
    R = (1/N) * (x_train') * x_train; 
    [U, Lambda] = eig(R);
    Lambdas = diag(Lambda);
    P = sum(Lambdas);
    percentage = 0.9;   % Percentage of eigenvalues we want to consider
    Z = x_train * U;    % We map initial features on orthogonal vectors
    aHatPCR = pinv(x_train) * y_train;
    F = length(aHatPCR);
    somma = 0;
    L = 0;
    while somma < percentage * P
        L = L + 1;
        somma = somma + Lambdas(L);
    end
    LambdaL = Lambda(1:L, 1:L);
    LambdasL = diag(LambdaL);
    UL = U(:, 1:L);
    aHatPCRL = (1/N) * UL * (inv(LambdaL)) * (UL') * (x_train') * y_train;
    
    % Result computing for both N and L features considered:
    y_train_hat_Nfeature = x_train * aHatPCR;   
    y_test_hat_Nfeature = x_test * aHatPCR;
    y_train_hat_Lfeature = x_train * aHatPCRL;
    y_test_hat_Lfeature = x_test * aHatPCRL;
    
    % Task 1
    figure, subplot(2,2,1)
    plot(y_train_hat_Nfeature, '--*'), hold on, grid on, plot(y_train)
    legend(['$\hat{y}$\_train\_N = ', num2str(F), ' features'], ...
        'y\_train', 'Location', 'northwest')
    title(['PCR: N = ', num2str(F), ' features TRAIN of F0 = ', ...
        num2str(F0(ii))])
    subplot(2,2,2)
    plot(y_train_hat_Lfeature, '--*'), hold on, grid on, plot(y_train)
    legend(['$\hat{y}$\_train\_L ', num2str(L), ' features'], ...
        'y\_train', 'Location', 'northwest')
    title(['PCR: N = ', num2str(L), ' features TRAIN of F0 = ', ...
        num2str(F0(ii))])
    
    % Task 2
    subplot(2,2,3)
    plot(y_test_hat_Nfeature, '--*'), hold on, grid on, plot(y_test)
    legend(['$\hat{y}$\_test\_N = ', num2str(F), ' features'], 'y\_test')
    title(['PCR: N = ', num2str(F), ' features TEST of F0 = ', ...
        num2str(F0(ii))])
    subplot(2,2,4)
    plot(y_test_hat_Lfeature, '--*'), hold on, grid on, plot(y_test)
    legend(['$\hat{y}$\_test\_L ', num2str(L), ' features'], 'y\_test')
    title(['PCR: N = ', num2str(L), ' features TEST of F0 = ', ...
        num2str(F0(ii))])
    
    % Task 3
    figure, subplot(2,2,1)
    hist(y_train - y_train_hat_Nfeature, 50), grid on
    title(['PCR: Prediction error on TRAIN for N = ', num2str(F), ...
        ' features. F0 = ', num2str(F0(ii))])
    subplot(2,2,2)
    hist(y_train - y_train_hat_Lfeature, 50), grid on
    title(['PCR: Prediction error on TRAIN for N = ', num2str(L), ...
        ' features. F0 = ', num2str(F0(ii))])
    
    % Task 4
    subplot(2,2,3)
    hist(y_test - y_test_hat_Nfeature, 50), grid on
    title(['PCR: Prediction error on TEST for N =  ', num2str(F), ...
        ' features. F0 = ', num2str(F0(ii))])
    subplot(2,2,4)
    hist(y_test - y_test_hat_Lfeature, 50), grid on
    title(['PCR: Prediction error on TEST for N = ', num2str(L), ...
        ' features. F0 = ', num2str(F0(ii))])
    
    % Task 5
    figure, plot(aHatPCR), grid on, hold on, plot(aHatPCRL, 'o')
    legend(['$\hat{a}$\_N with N = ', num2str(F), ' features'], ...
        ['$\hat{a}$\_L with L = ', num2str(L), ' features'], ...
        'Location', 'northwest')
    title(['PCR: Coefficients for F0 = ', num2str(F0(ii))])
    
    figure, subplot(2,1,1)
    plot(y_train, y_train_hat_Nfeature, 'o'), grid on
    title(['PCR: Regression for N = ', num2str(F), ...
        ' features and F0 = ', num2str(F0(ii))])
    xlabel('y\_train'), ylabel('$\hat{y}$\_train')
    subplot(2,1,2)
    plot(y_train, y_train_hat_Lfeature, 'o'), grid on
    title(['PCR: Regression for N = ', num2str(L), ...
        ' features and F0 = ', num2str(F0(ii))])
    xlabel('y\_train'), ylabel('$\hat{y}$\_train')
end