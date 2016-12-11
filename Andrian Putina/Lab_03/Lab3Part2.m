close all
clear all
% Lab 3 - Execution

load('arrhythmiaCleaned.mat');

class_id = arrhythmiaCleaned(:,end);
y = arrhythmiaCleaned(:,1:end-1);

class1 = class_id(:) == 1;
class2 = class_id(:) == 2;

y1 = y(class1,:);
y2 = y(class2,:);

x1 = mean(y1);
x2 = mean(y2);

%COMPUTATION

xmeans = [x1;x2];
eny = diag(y*transpose(y));
enx = diag(xmeans*transpose(xmeans));
dotprod = y * transpose(xmeans);
%each y(n) and each x
[U,V]=meshgrid(enx,eny);
dist2=U+V-2*dotprod;


[dummy, previsione] = min(dist2.');
previsione = previsione';

verinegativi = 0;
falsipositivi = 0;
veripositivi = 0;
falsinegativi = 0;
arrhythmia_lastcol = arrhythmiaCleaned(:,end);
for i = 1 : length(arrhythmiaCleaned) 
    if (arrhythmia_lastcol(i) == 1)
        if (arrhythmia_lastcol(i) - previsione(i) == 0)
            verinegativi = verinegativi + 1;
        else 
            falsipositivi = falsipositivi + 1;
        end
    else
        if (arrhythmia_lastcol(i) - previsione(i) == 0) 
            veripositivi = veripositivi + 1;
        else 
            falsinegativi = falsinegativi + 1;
        end
    end
end

specificity = verinegativi / (verinegativi + falsipositivi); %true negative
sensitivity = veripositivi / (veripositivi + falsinegativi); % true positive
falsealarm = falsipositivi / (verinegativi + falsipositivi);
misseddetection = falsinegativi / (veripositivi + falsinegativi);

%BAYES

pi1 = size(y1,1)/ size(y,1);
pi2 = size(y2,1)/ size(y,1);

meany = mean(y);
stdy = std(y);

y = y - meany;
y = y ./ stdy; 


% perform PCR
N = size(y,1);
R = 1/N * transpose(y) * y;
[U, A] = eig(R);
% DINAMICA L
total_eig = sum(diag(A));
percentage_thresh = 100 * total_eig;

sum_diag = 0;
L = 257;
% while sum_diag < percentage_thresh
%     sum_diag = A(L,L) + sum_diag; 
%     L = L+1;
% end

U_L = U(:, 1:L);
Z = y * U_L;
% Z = (1/sqrt(N)) * Z * A^(-1/2);

for i=1:size(Z)
    Z_norm(i,:) = (Z(i,:) - mean(Z))./std(Z);
end

Z=[];
Z = Z_norm;


z1 = Z(class1,:);
z2 = Z(class2,:);
w1 = mean(z1,1);
w2 = mean(z2,1);

xmeansB = [w1;w2];
enyB = diag(Z*transpose(Z));
enxB = diag(xmeansB*transpose(xmeansB));
dotprodB = Z * transpose(xmeansB);
%each y(n) and each x
[UB,VB]=meshgrid(enxB,enyB);
dist2B=UB+VB-2*dotprodB;

var_patients = var(Z);

dist2B_bis(:,1)= dist2B(:,1)- 2 * log(pi1);
dist2B_bis(:,2)= dist2B(:,2)- 2 * log(pi2);

[dummy, previsione2] = min(dist2B_bis.');
previsione2 = previsione2';



verinegativiB = 0;
falsipositiviB = 0;
veripositiviB = 0;
falsinegativiB = 0;
arrhythmia_lastcol = arrhythmiaCleaned(:,end);
for i = 1 : length(arrhythmiaCleaned) 
    if (arrhythmia_lastcol(i) == 1)
        if (arrhythmia_lastcol(i) - previsione2(i) == 0)
            verinegativiB = verinegativiB + 1;
        else 
            falsipositiviB = falsipositiviB + 1;
        end
    else
        if (arrhythmia_lastcol(i) - previsione2(i) == 0) 
            veripositiviB = veripositiviB + 1;
        else 
            falsinegativiB = falsinegativiB + 1;
        end
    end
end

specificityB = verinegativiB / (verinegativiB + falsipositiviB); %true negative
sensitivityB = veripositiviB / (veripositiviB + falsinegativiB); % true positive
falsealarmB = falsipositiviB / (verinegativiB + falsipositiviB);
misseddetectionB = falsinegativiB / (veripositiviB + falsinegativiB);











