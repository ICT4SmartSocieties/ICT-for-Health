clear all
close all
clc

load('ckd3.mat');

%% ----------- Data Preparation ------------ %%

keylist={'normal','abnormal','present','notpresent','yes','no','good','poor','ckd','notckd','?',''};
keymap=[0,1,0,1,0,1,0,1,2,1,NaN,NaN];

ckd=chronickidneydisease;


% adjusting the format of the input data
[kR,kC] = size(ckd);
b=[];
for kr=1:kR
    for kc=1:kC
        c=strtrim(ckd(kr,kc)); % remove blanks
        check=strcmp(c,keylist);
        if sum(check)==0
            b(kr,kc)=str2num(ckd{kr,kc}); % from text to numeric
        else
            ii=find(check==1);
            b(kr,kc)=keymap(ii); % use the lists
        end
    end
end

b=b(:,1:end-1); % I mistakenly considered an empty column in the import phase

[N,F]=size(b);

X=b(:,1:end-1); % we don't consider the last column, that stores classification evaluated by the doctor
classes=b(:,end);



%% ---------- Hierachical Clustering ---------- %%

D=pdist(X); % algorithm that evaluates the distance between each 
            % measurement stored in the matrix
D_matrix=squareform(D); % the output of pdist is a row vector. 'squareform' 
                 % transforms it into a matrix of distances d(i,j)
                 
Z=linkage(D); % 

K=2; % selecting the number of clusters we want to consider

    % we create the hierachical tree
T=cluster(Z,'maxclust',K);
    % The output matrix Z contains cluster information. Z has size m-1 by 3
    % where m is the number of observations in the original data. Each newly-formed
    % cluster, corresponding to Z(i,:), is assigned the index m+i, where m is
    % the total number of initial leaves. Z(i,1:2) contains the indices of
    % the two component clusters which form cluster m+i. There are m-1 higher
    % clusters which correspond to the interior nodes of the output
    % clustering tree. Z(i,3) contains the corresponding linkage distances
    % between the two clusters which are merged in Z(i,:).

p=0;
figure
dendrogram(Z,p) 

% we compare the clustering with the classification given by doctors
perc_true=length(find(T==classes))/N; % 0.8025
perc_false=length(find(T~=classes))/N; % 0.1975 (1-perc_true)


% trying to evaluate the sum of the squared error, that measures the
% performance of the clustering algorithm
w1=X(find(classes==1),:);
w2=X(find(classes==2),:);
m_k(1,:)=mean(w1,1);
m_k(2,:)=mean(w2,1);

SSE=0;
for i=1:size(w1,1)
    error_1(i)=norm(w1(i,:)-m_k(1,:)).^2;
    SSE=SSE+error_1(i);
end

for i=1:size(w2,1)
    error_2(i)=norm(w2(i,:)-m_k(2,:)).^2;
    SSE=SSE+error_2(i);
end

% SSE doesn't work hihihii



%% --------- Classification Tree ---------- %%

tc=fitctree(X,classes); % function that generates the classification tree

view(tc); % decision tree explained in command window
view(tc,'Mode','graph'); % graphical representation of the decision tree

% implementation of the decision tree
for i=1:N
    if X(i,15)<13.05
        if X(i,16)<44.5
            ct_classes(i)=2;
        else
            ct_classes(i)=1;
        end
    else
        if X(i,3)<1.0175
            ct_classes(i)=2;
        else
            if X(i,4)<0.5
                ct_classes(i)=1;
            else
                ct_classes(i)=2;
            end
        end
    end
end

ct_classes=ct_classes';

perc_true_ct=length(find(ct_classes==classes))/N; % 0.9275
perc_false_ct=length(find(ct_classes~=classes))/N; % 0.0725 (1-perc_true)