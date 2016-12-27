%************************************************************
%   Created by Andrian Putina                               *
% Lab 4 - Hard K-Means Algorithm with probability variation *
%************************************************************

close all
clear all
%Lab 4 - HardK Means 

addpath('../functions/');
load('arrhythmiaCleaned.mat');

data = arrhythmiaCleaned(:,1:end-1);
class_id = arrhythmiaCleaned(:,end);
norm_data = normalize_matrix(data);
y = arrhythmiaCleaned(:,1:end-1);

class1 = class_id(:) == 1;
class2 = class_id(:) == 2;

y1 = norm_data(class1,:);
y2 = norm_data(class2,:);

x1 = mean(y1);
x2 = mean(y2);

k=2;
n_features = size(norm_data,2);
N_pacients = size(norm_data,1);

Nk1=0;
Nk2=0;
Wn1=[];
Wn2=[];
assigned_class=zeros(N_pacients,1);

K = 2;
var_k = ones(1,K);
pi_k =  1/K*ones(1,K);
patient_cluster = zeros(N_pacients,1);

count=0;
flag = 1;
tic,
while(flag & toc < 10)    
    
    flag = 0;
        
    for ii=1:N_pacients
        
        R_k = (pi_k(1)/((2*pi*var_k(1))^(N_pacients/2))) * exp(-norm(norm_data(ii,:)-x1)^2/(2*var_k(1)));
        R_j = (pi_k(2)/((2*pi*var_k(2))^(N_pacients/2))) * exp(-norm(norm_data(ii,:)-x2)^2/(2*var_k(2)));
        
        if (R_k < R_j)
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
    
    pi_k(1)=Nk1/N_pacients;
    pi_k(2)=Nk2/N_pacients;
    
    x1 = 1/Nk1 * sum(Wn1,1);
    x2 = 1/Nk2 * sum(Wn2,1);
    
    var_k(1) = sum(norm(Wn1-x1)^2)/(Nk1-1)/n_features;
    var_k(2) = sum(norm(Wn2-x2)^2)/(Nk2-1)/n_features;
       
    Nk1=0;
    Nk2=0;
    Wn1=[];
    Wn2=[];
    
    count = count+1;
end

[specificity, sensitivity, falsealarm, missdetection] = check_detections(assigned_class, class_id);