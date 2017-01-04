clear all
close all
clc

load('arrhythmia.mat')

%% -----Data Preparation----------- %

A=arrhythmia;

A(:, find(sum(abs(A)) == 0)) = []; % we erase the zero columns

class_id=A(:,end); % last vector of the matrix
y=A;
y(:,end)=[]; % we put in y all the features but the last one
[N,F]=size(y);

%normalizing y
mean_y=mean(y,1);
stdv_y=std(y,1);

o=ones(N,1);% o is a column vector 
y=(y-o*mean_y)./(o*stdv_y);% y is normalized

mean_y=mean(y,1); % checking that y matrix is properly normalized
var_y=var(y,1);


% we make a list of classes
classes = sort(unique(class_id)); 
C=length(classes);


for i=1:max(class_id)
    N_classes(i)=sum(class_id==i); % vector that stores the n. of occurrences for each class
    xmeans(i,:)=mean(y(find(class_id==i),:),1);
end

%% -------- Minimum distance criterion ----------- %

eny=diag(y*y'); % |y(n)|^2
enx=diag(xmeans*xmeans'); % |x1|^2 and |x2|^2
dotprod=y*xmeans'; % matrix with the dot product between each y(n) and each x
[U,V]=meshgrid(enx,eny);
dist=U+V-2*dotprod; % |y(n)|^2+|x(n)|^2-2y(n)x(k)=|y(n)-x(k)|^2

[M,decision]=min(dist,[],2); % taking the decision
%'decision' is an array of length N with the corresponding closest region
% for each element (patient)

p_true=length(find(decision==class_id))/N; % 0.6725
p_false=length(find(decision~=class_id))/N; % 0.3274 (1-p_true)




%% ------- Bayes criterion ------------------------%

% define the probabilities for each region
for i=1:max(class_id)
    pis(i)=N_classes(i)/N;
end

%---perforiming PCA----%

R_y=y'*y/N;
[U, E] = eig(R_y);

P = sum(diag(E));
percentage = 0.999; % we set the percentage of information that we want to keep
new_P = percentage * P; 

cumulative_P = cumsum(diag(E)); % function that evaluates the cumulative 
                                % sum of each element of the diagonal of A 
L = length(find(cumulative_P<new_P)); % determines the first L features 
                                % that contribut to obtain new_P amount 
                                % of "information"
                                
U_L = U(:,1:L); % we only consider the first L features

Z = y * U_L;
mean_Z=mean(Z,1); % Z is zero mean
Z=Z./(o*sqrt(var(Z)));  % we normalize Z

for i=1:max(class_id)
    wmeans(i,:)=mean(Z(find(class_id==i),:),1);
end

enZ=diag(Z*Z'); % |Z(n)|^2
enW=diag(wmeans*wmeans'); % |w1|^2 and |w2|^2
dotprod_2=Z*wmeans'; % matrix with the dot product between each Z(n) and each w
[U2,V2]=meshgrid(enW,enZ);
dist_z=U2+V2-2*dotprod_2; % |y(n)|^2+|x(n)|^2-2y(n)x(k)=|y(n)-x(k)|^2

onevar=ones(N,1);
bayes_dist=dist_z-2*onevar*log(pis); % evaluating the bayesian distance

[M,decision_bayes]=min(bayes_dist,[],2); % taking the decision

p_true_bayes=length(find(decision_bayes==class_id))/N; % 0.9424
p_false_bayes=length(find(decision_bayes~=class_id))/N; % 0.0575


