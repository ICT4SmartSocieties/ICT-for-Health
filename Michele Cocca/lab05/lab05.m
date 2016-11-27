%%% LAB05 %%%
close all
clear all
clc

load('ckd.mat');
ckd = chronickidneydisease;
%% define the vocabualry
keylist={'normal','abnormal','present','notpresent','yes','no','good','poor','ckd','notckd','?',''};
keymap=[0,1,0,1,0,1,0,1,2,1,NaN,NaN];

%% convertion of the strng whith values
[kR,kC] = size(ckd);
b=[];

for kr=1:kR
    for kc=1:kC
        c=strtrim(ckd(kr,kc)); %remove blanks
        check=strcmp(c,keylist);
        if sum(check)==0
            b(kr,kc)=str2num(ckd{kr,kc});% from text to numeric
        else
            ii=find(check==1);
            b(kr,kc)=keymap(ii);% use the lists
        end
    end
end
ckdCol = b(:,kC);
b(:,kC) = [];


%% perform clustering
d = pdist(b);
s = squareform(d);
Tree = linkage(d);
c = cluster(Tree,'maxclust',2);

figure
p=0;% p=0 means that all the leaves must be included in the plotted tree
dendrogram(Tree,p);
figure
nn=[1:kR];
plot(nn,c,'o'),grid on
xlabel('i')
ylabel('cluster for the i-th row of X')

%% perform classification with ckd and notckd
b(:,kC) = ckdCol;
tc = fitctree(b,c);
view(tc,'Mode','graph')
figure