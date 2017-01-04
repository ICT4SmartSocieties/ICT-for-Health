clear all
close all

load('updrs.mat')

A=parkinsonsupdrs;

num_patients=max(A(:,1));

new_matrix=ones(1,22); % we define the matrix in which we insert grouped values

for ind=1:num_patients
    sub_rows=find(A(:,1)==ind); % indexes of the matrix corresponding to the 'ind' patient
    pat=A(sub_rows, :); % sub_matrix of 'ind' patient
    time=sort(abs(fix(pat(:,4)))); % we define an array of the days in which measurements were taken for patient 'ind'
    % we had to take the integer part of that values and make the absolute
    % value because there were some negative time values. Then I ordered
    % them by increasing order
    time=unique(time); % we delete duplicates in the array of days
    for i=1:length(time) %we iterate within the days
        day=time(i);
        time_rows=find(A(:,1)==ind & abs(fix(A(:,4)))==day); % we find the rows with all the measurements taken on patient 'ind' in the day 'day'
        pat_time=A(time_rows, :); % we build a matrix of the measurements of patient 'ind' on day 'day'
        pat_time(:,4)=abs(fix(pat_time(:,4))); % we adjust the time column in order to have integer day indexes
        mean_row=mean(pat_time,1); 
        new_matrix=[new_matrix;mean_row];
    end
end

pazienti=new_matrix(2:end,:);


%-------  we build data_train and data_test matrices-----

% data_train=ones(1,22);
% data_test=ones(1,22);
% for i=1:length(new_matrix(:,1))
%     if new_matrix(i,1)<=36 
%         data_train=[data_train;new_matrix(i,:)];
%     else
%         data_test=[data_test;new_matrix(i,:)];
%     end
% end
% 
% data_train=data_train(2:end,:);
% data_test=data_test(2:end,:);

data_train = pazienti(pazienti(:,1)<37,:);
data_test = pazienti(pazienti(:,1)>36,:);


% ---- applying normalization to data_train matrix

m_data_train=mean(data_train,1);
v_data_train=var(data_train,1);
s_v_data_train=sqrt(v_data_train);

% data_train_norm=ones(length(data_train(:,1)),22);
% for i=1:length(data_train(:,1))
%     data_train_norm(i,1:4)=data_train(i,1:4);
%     for f=5:22
%         data_train_norm(i,f)=(data_train(i,f)-m_data_train(f))/s_v_data_train(f);
%     end
% end

o = ones(size(data_train,1),1);
data_train_norm = data_train;
data_train_norm(:,5:end) = (data_train(:,5:end) - o*m_data_train(:,5:end)) ./ (o*s_v_data_train(:,5:end));

mean_train_norm = mean(data_train_norm,1); % verify that normalization is effective 
var_train_norm = var(data_train_norm,1); 


% ---- applying normalization to data_test matrix

% data_test_norm=ones(length(data_test(:,1)),22);
% for i=1:length(data_test(:,1))
%     data_test_norm(i,1:4)=data_test(i,1:4);
%     for f=5:22
%         data_test_norm(i,f)=(data_test(i,f)-m_data_train(f))/s_v_data_train(f);
%     end
% end

o = ones(size(data_test,1),1);
data_test_norm = data_test;
data_test_norm(:,5:end) = (data_test(:,5:end) - o*m_data_train(:,5:end)) ./ (o*s_v_data_train(:,5:end));

mean_test_norm = mean(data_train_norm,1); % verify that normalization is effective
var_test_norm = var(data_train_norm,1);

% ---- PERFORM REGRESSION ----------

F0=5; % we have to try with feature 7 and feature 5

y_train=data_train_norm(:,F0);
X_train=data_train_norm;
X_train(:,F0)=[];

y_test=data_test_norm(:,F0);
X_test=data_test_norm;
X_test(:,F0)=[];




% ----------------------MSE------


X_T=X_train(:,5:end).';

a_hat=inv(X_T*X_train(:,5:end))*X_T*y_train;

yhat_train=X_train(:,5:end)*a_hat;
yhat_test=X_test(:,5:end)*a_hat;

error_test=y_test-yhat_test;
error_train=y_train-yhat_train;

figure
plot(yhat_train)
hold on
plot(y_train)
hold off
title('MSE: yhat train vs y train')
legend('yhat train','y train')

figure
plot(yhat_test)
hold on
plot(y_test)
hold off
title('MSE: yhat test vs y test')
legend('yhat test','y test')

figure
hist(error_test,50) % for the feature 5 the error is very variable, that means that feature 5 is less correlated to others
% for the feature 7 the error is distributed according something that is
% like a gaussian
title('MSE: histogram of the error distribution')




%--------- Gradient Algorithm --------------

rng('default') % allows to have the same random vector each time we run the script
a_i = rand(17,1); % generates a 21x1 matrix (a vector) of uniformly distributed random numbers
gamma=0.000001; % learning coefficient (usually 0.01/0.001/0.0001)
epsilon=0.00001; % stopping rule: a_hat(i+1)-a_hat(i)<epsilon
a_old=zeros(17,1); 
n=0;
a_story=a_i';

while (norm(a_i - a_old) > epsilon)
    grad= - 2 * X_T * y_train + 2 * X_T * X_train(:,5:end) * a_i;
    a_story=[a_story;a_i'];
    a_old=a_i;
    a_i = a_i - (gamma * grad);
    n=n+1;
end

yhat_ga_test=X_test(:,5:end)*a_i;
error_ga_test=y_test-yhat_ga_test;

figure
plot(yhat_ga_test)
hold on
plot(y_test)
hold off
title('GRADIENT ALGORITHM: yhat test vs y test')
legend('yhat test','y test')

figure
hist(error_ga_test,50)
title('GRADIENT ALGORITHM: histogram of the error distribution')





%-------steepest descent algorithm----------%


rng('default') % allows to have the same random vector each time we run the script
a_i_sd = rand(17,1); % generates a 21x1 matrix (a vector) of uniformly distributed random numbers
gamma=0.000001; % learning coefficient (usually 0.01/0.001/0.0001)
epsilon=0.00001; % stopping rule: a_hat(i+1)-a_hat(i)<epsilon
a_old_sd=zeros(17,1); 
n=0;
a_story=a_i';

while (norm(a_i_sd - a_old_sd) > epsilon)
    grad_sd= - 2 * X_T * y_train + 2 * X_T * X_train(:,5:end) * a_i_sd;
    hessian = 4 * X_T * X_train(:,5:end);
    a_old_sd = a_i_sd;
    a_i_sd = a_i_sd - ((norm(grad_sd)^2 * grad_sd)/(grad_sd.' * hessian * grad_sd));
end

yhat_sd_test=X_test(:,5:end)*a_i_sd;
error_sd_test=y_test-yhat_sd_test;

figure
plot(yhat_sd_test)
hold on
plot(y_test)
hold off
title('STEEPEST DESCENT: yhat test vs y test')
legend('yhat test','y test')

figure
hist(error_sd_test,50)
title('STEEPEST DESCENT: histogram of the error distribution')

    