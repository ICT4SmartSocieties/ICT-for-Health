close all
clear all
clc
%% Loading
load('arrhythmia.mat')

%% remove column that contains only zeros
[N,F] = size(arrhythmia);
columnToRemove=sum(arrhythmia(:,1:F))==0;
arrhythmia(:,columnToRemove) = [];
[R_arr, C_arr] = size(arrhythmia);

%change in the last column all value > 1 with 2
lastColumn = arrhythmia(:,end);
arrhythmia(:,end) = lastColumn;

%% Define y and class_id
y=arrhythmia(:,1:end-1);
meany = mean(y);
vary = var(y);
o = ones(size(y,1),1);
m_meany = o*meany;
m_vary = o*vary;
y = (y-m_meany) ./sqrt(m_vary);
class_id = arrhythmia(:,end);

%y1 contains patient with class id=1; y2->lastid =2
[R,C] = size(y);
y1=[]; y2=[];

okV1 = class_id==1;
okV2 = class_id==2;
y1 = y(okV1,:);
y2 = y(okV2,:);

%x1 and x2, means of each column (computed only on the significant data)
x1 = mean(y1);
x2 = mean(y2);

%apply the minimum distance criterion for each rows of y
xmeans=[x1;x2];% matrix with x1 and x2
eny=diag(y*y');% |y(n)|^2
enx=diag(xmeans*xmeans');% |x1|^2 and |x2|^2
dotprod=y*xmeans';% matrix with the dot product between
% each y(n) and each x
[U,V]=meshgrid(enx,eny);
dist=U+V-2*dotprod;%|y(n)|^2+|x(k)|^2-2y(n)x(k)=
%=|y(n)-x(k)|^2

%Measure the T/F negative/positive
[M,computed_class_id] = min(dist,[],2);
%1 health, 2seek
true_negative =0;
false_positive=0;
false_negative=0;
true_positive =0;
for i=1:R_arr
   if class_id(i) == 1 && computed_class_id(i) == 1
       true_negative=true_negative+1;
   elseif class_id(i) == 1 && computed_class_id(i) == 2
       false_positive = false_positive +1;
   elseif class_id(i) == 2 && computed_class_id(i) == 1
       false_negative = false_negative + 1;
   else
       true_positive = true_positive + 1;
   end      
end

total_seek = true_positive + false_negative;
total_health = true_negative + false_positive;

res1 = [true_negative/total_health
       false_positive/total_health
       false_negative/total_seek
       true_positive/total_seek
       ];
   
 
%% BAYES and REDUCED MATRIX
%use the bayes criterion to measerue pi1 (number of patient without arr)
%and pi2 the number of patient with arrhytmia
sorted_class_id = sortrows(class_id);
number_of_health = sorted_class_id == 1;
[N_patient, columns] = size(class_id);
pi1 = sum(number_of_health)/N_patient;
pi2 = (N_patient-sum(number_of_health))/N_patient;

%Reduced Matrix
%measure the covariance matrix Ry of y and takes the largest eigenval
Ry = cov(y);
[eVect, eVal] = eig(Ry);
eVal = diag(eVal);
total = sum(eVal);
s=0;
i=size(eVal,1);
%%%%%%chech cumsum()
while s <0.99
    s = s + eVal(i)/total;
    i=i-1;
end 
F1 = size(eVal,1) - i;

%transform y in z=y*UF1 cmatrix and takes the cov(Z)
UF1 = eVect(1:end, end-F1+1:end);
z=y*UF1;

meanz = mean(z);
varz = var(z);
for i=1:size(z,2)
    z(:,i)=(z(:,i) - meanz(i))/sqrt(varz(i));
end
Rz = cov(z);

%create z1 and z2 as y1 and y2
okV1 = class_id==1
okV2 = class_id==2
z1 = z(okV1,:);
z2 = z(okV2,:);
w1 = mean(z1);
w2 = mean(z2);

%apply the minimum distance criterion for each rows of y
wmeans=[w1;w2];% matrix with x1 and x2
enz=diag(z*z');% |y(n)|^2
enw=diag(wmeans*wmeans');% |x1|^2 and |x2|^2
dotprod2=z*wmeans';% matrix with the dot product between
% each y(n) and each x
[U2,V2]=meshgrid(enw,enz);
dist2=U2+V2-2*dotprod2;%|y(n)|^2+|x(k)|^2-2y(n)x(k)=
%=|y(n)-x(k)|^2
distEst(:,1) = dist2(:,1) - (2* 1 *log(pi1));
distEst(:,2) = dist2(:,2) - (2* 1 *log(pi2));
[M,est_class_id2] = min(distEst,[],2);
%1 health, 2seek
true_negative2 =0;
false_positive2=0;
false_negative2=0;
true_positive2=0;
for i=1:size(est_class_id2,1)
   if class_id(i) == 1 && est_class_id2(i) == 1
       true_negative2 = true_negative2 + 1;
       
   elseif class_id(i) == 1 && est_class_id2(i) == 2
       false_positive2 = false_positive2 +1;
       
   elseif class_id(i) == 2 && est_class_id2(i) == 1
       false_negative2 = false_negative2 + 1;
   
   else
       true_positive2 = true_positive2 + 1;
   end      
end
total_seek2 = true_positive2 + false_negative2;
total_health2 = true_negative2 + false_positive2;

res2 = [true_negative2/total_health2
       false_positive2/total_health2
       false_negative2/total_seek2
       true_positive2/total_seek2
       ];
   
total = [res1, res2]
