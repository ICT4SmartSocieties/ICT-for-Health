close all
clear all
clc
%% Loading
load('arrhythmia.mat')

%% remove column that contains only zeros
[N,F] = size(arrhythmia);
columnToRemove= sum(arrhythmia(:,1:F))==0;
arrhythmia(:,columnToRemove) = [];

%% Define y and class_id
y=arrhythmia(:,1:end-1);
meany = mean(y);
vary = var(y);
o = ones(size(y,1),1);
m_meany = o*meany;
m_vary = o*vary;
y = (y-m_meany) ./sqrt(m_vary);
class_id = arrhythmia(:,end);
classes = 16;
[classes_numbers,] = [1:classes]';

%y1 contains patient with class id=1; y2->lastid =2
y_grouped=[];
row_per_class=[];
xmeans=[];
x=[];
for i=1:size(classes_numbers,1)
   okV = class_id==classes_numbers(i);
   xmeans = [xmeans; mean(y(okV,:))];
   row_per_class=[row_per_class; size(y(okV,:),1)];
end

%apply the minimum distance criterion for each rows of y
eny=diag(y*y');% |y(n)|^2
enx=diag(xmeans*xmeans');% |x1|^2 and |x2|^2
dotprod=y*xmeans';% matrix with the dot product between
% each y(n) and each x
[U,V]=meshgrid(enx,eny);
dist=U+V-2*dotprod;%|y(n)|^2+|x(k)|^2-2y(n)x(k)=
%=|y(n)-x(k)|^2
%Measure the T/F negative/positive

[M,computed_class_id] = min(dist,[],2);
pi= row_per_class/sum(row_per_class);
correct_predictions=zeros(classes,1); 

% R = class_id | C = estimated_class_id
hitV = class_id - computed_class_id ==0;
cp = class_id(hitV,:);
for ii=1:classes
    cpp(ii) = sum(cp == ii)
end
for i=1:size(class_id,1)
   if class_id(i) == computed_class_id(i)
        correct_predictions(class_id(i))=correct_predictions(class_id(i))+1; 
   end
end

precision2=0;
for i=1:size(row_per_class,1)
    if row_per_class(i) > 0
        precision2 = precision2 +(correct_predictions(i)/row_per_class(i) * pi(i));
    end
end
r=precision2;
%% BAYES and REDUCED MATRIX


Ry = cov(y);
[eVect, eVal] = eig(Ry);
eVal = diag(eVal);
total = sum(eVal);
s=0;
i=size(eVal,1);
while s <=0.9999
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

wmeans=[];
row_per_class_b=[];
for i=1:classes
   okV = class_id==classes_numbers(i);
   wmeans = [wmeans; mean(z(okV,:))];
   row_per_class_b=[row_per_class_b; size(z(okV,:),1)];
end

%apply the minimum distance criterion for each rows of y
enz=diag(z*z');% |y(n)|^2
enw=diag(wmeans*wmeans');% |x1|^2 and |x2|^2
dotprod2=z*wmeans';% matrix with the dot product between
% each y(n) and each x
[U2,V2]=meshgrid(enw,enz);
dist2=U2+V2-2*dotprod2;%|y(n)|^2+|x(k)|^2-2y(n)x(k)=
%=|y(n)-x(k)|^2
for i=1:classes
    distEst(:,i) = dist2(:,i) - (2* 1 *log(pi(i)));
end

[M,est_class_id2] = min(distEst,[],2);
correct_predictions2=zeros(classes,1);
for i=1:size(class_id,1)
   if class_id(i) == est_class_id2(i)
        correct_predictions2(class_id(i))=correct_predictions2(class_id(i))+1; 
   end
end

precision2=0;
for i=1:size(row_per_class,1)
    if row_per_class(i) > 0
        precision2 = precision2 +(correct_predictions2(i)/row_per_class(i) * pi(i));
    end
end
r=[r precision2]
 
