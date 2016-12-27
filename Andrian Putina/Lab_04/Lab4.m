%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Created by Andrian Putina    %
% Lab 4 - Hard K-Means Algorithm %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clear all
%Lab 4 - HardK Means 

addpath('../functions/');
load('arrhythmiaCleaned.mat');

data = arrhythmiaCleaned(:,1:end-1);
class_id = arrhythmiaCleaned(:,end);
norm_data = normalize_matrix(data);

class1 = class_id(:) == 1;
class2 = class_id(:) == 2;

y1 = norm_data(class1,:);
y2 = norm_data(class2,:);

x1 = mean(y1);
x2 = mean(y2);

k=2;
n_features = size(norm_data,2);
N_pacients = size(norm_data,1);

% x1 = normalize_matrix(randn(1,n_features));
% x2 = normalize_matrix(randn(1,n_features));
% pi1 = 1/k;
% pi2 = 1/k;
% 
Nk1=0;
Nk2=0;
Wn1=[];
Wn2=[];
assigned_class=zeros(N_pacients,1);

count=0;
flag = 1;
while(flag)    
    
    flag = 0;
    
    for ii=1:N_pacients
        
        dist1 = norm(x1 - norm_data(ii,:));
        dist2 = norm(x2 - norm_data(ii,:));

        if (dist1 < dist2)
            Nk1 = Nk1+1;
            Wn1 = [Wn1; norm_data(ii,:)];
            if assigned_class(ii)==1
            else
                assigned_class(ii)=1;
                flag = 1;
            end
            
        else
            Nk2 = Nk2+1;
            Wn2 = [Wn2; norm_data(ii,:)];
            if assigned_class(ii)==2
            else
                assigned_class(ii)=2;
                flag=1;
            end
        end
    end
    
%     Wn1 = normalize_matrix(Wn1);
%     Wn2 = normalize_matrix(Wn2);
    
    x1 = 1/Nk1 * sum(Wn1,1);
    x2 = 1/Nk2 * sum(Wn2,1);
       
    Nk1=0;
    Nk2=0;
    Wn1=[];
    Wn2=[];
    
    count = count+1;
end
[specificity, sensitivity, falsealarm, missdetection] = check_detections(assigned_class, class_id);


y = arrhythmiaCleaned(:,1:end-1);

x1 = mean(y1);
x2 = mean(y2);

count=0;
flag = 1;
while(flag)    
    
    flag = 0;
    
    xmeans = [x1;x2];
    eny = diag(y*transpose(y));
    enx = diag(xmeans*transpose(xmeans));
    dotprod = y * transpose(xmeans);
    %each y(n) and each x
    [U,V]=meshgrid(enx,eny);
    dist2=U+V-2*dotprod;

    [dummy, previsione] = min(dist2.');
    previsione = previsione';
    
    for ii=1:N_pacients
        
        if (previsione(ii) == 1)
            Nk1 = Nk1+1;
            Wn1 = [Wn1; norm_data(ii,:)];
            if assigned_class(ii)==1
            else
                assigned_class(ii)=1;
                flag = 1;
            end
            
        else
            Nk2 = Nk2+1;
            Wn2 = [Wn2; norm_data(ii,:)];
            if assigned_class(ii)==2
            else
                assigned_class(ii)=2;
                flag=1;
            end
        end
    end
    
%     Wn1 = normalize_matrix(Wn1);
%     Wn2 = normalize_matrix(Wn2);
    
    x1 = 1/Nk1 * sum(Wn1,1);
    x2 = 1/Nk2 * sum(Wn2,1);
       
    Nk1=0;
    Nk2=0;
    Wn1=[];
    Wn2=[];
    
    count = count+1;
end

[specificity2, sensitivity2, falsealarm2, missdetection2] = check_detections(assigned_class, class_id);





