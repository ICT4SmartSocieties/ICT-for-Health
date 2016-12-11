close all
clear all
% Lab 5 - Prepare Data

load('a.mat');

keylist={'normal','abnormal','present','notpresent','yes','no','good','poor','ckd','notckd','?',''};     
keymap=[0,1,0,1,0,1,0,1,2,1,NaN,NaN];
%%
for kr = 1:size(chronickidneydisease,1)
    for kc = 1:size(chronickidneydisease,2)

        c = strtrim(chronickidneydisease(kr,kc));
        check=strcmp(c,keylist);% check(i)=1 if c==keylist(i)

        if sum(check)==0
            b(kr,kc)=str2num(chronickidneydisease{kr,kc});% from text to numeric
        else
            ii=find(check==1);
            b(kr,kc)=keymap(ii);% use the lists
        end;
        
    end
end

%%
b=b(:,1:end-1);

D = pdist(b(:,1:end-1));    
tree = linkage(D);
c = cluster(tree, 'maxclust', 2);
figure
dendrogram(tree,0)








