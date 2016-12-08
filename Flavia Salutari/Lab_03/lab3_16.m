close all
clear all
clc

load('arrhythmia.mat');

arrhythmia(:, find(sum(abs(arrhythmia)) == 0)) = []; %remove null columns

class_id = arrhythmia(:,end);
classes = sort(unique(class_id)); 
y = arrhythmia(:,1: end-1);

meany = mean(y);
[N,F]=size(y);
mean_y = mean(y);
std_y = std(y);
o=ones(N,1);% o is a column vector 
y=(y-o*mean_y)./(o*std_y);% y is normalized

pis=zeros(1,max(classes));
for ii=1:16
    Nx=sum(class_id==ii);
    pis(ii)=Nx/N;
    indexes = find(class_id == ii);
    x16(ii,:) = mean(y(indexes,:),1);
end

eny=diag(y*y'); 
enx=diag(x16*x16'); 
dotprod = y*x16'; 
[U,V]=meshgrid(enx,eny); 
dist16 =U+V-2*dotprod;

[dummy,decz]=min(dist16,[],2);


false_detected=sum(decz~=class_id)/N; %0.3274
true_detected=sum(decz==class_id)/N; %0.6726


%% Bayes

R=y'*y/N; % R is F x F 
[U, A] = eig(R);
total_eig = sum(diag(A));
perc = 0.999;
percentage_thresh = perc * total_eig;

d = diag(A);
dcum = cumsum(d);
L = length(find(dcum<percentage_thresh));


K = 1;
U_L = U(:,K:L);
A_L = A(K:L,K:L);

Z = y * U_L;
Z=Z./(o*sqrt(var(Z)));


for i=1:length(classes)
    ii = classes(i);
    indexes = find(class_id == ii);
    w16(ii,:) = mean(Z(indexes,:),1);
end

eny2=diag(Z*Z'); 
enx2=diag(w16*w16');
dotprod = Z*w16';
[U2,V2]=meshgrid(enx2,eny2); 
dist16_b=U2+V2-2*dotprod;

dist16_b=dist16_b-2*o*log(pis);


[dummy,decz_b]=min(dist16_b,[],2);


false_detected_b=sum(decz_b~=class_id)/N; % 0.0553
true_detected_b=sum(decz_b==class_id)/N; % 0.9447

