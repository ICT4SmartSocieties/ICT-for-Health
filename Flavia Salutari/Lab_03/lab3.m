close all
clear all
clc

load('arrhythmia.mat');
rows_arrythmic = find(arrhythmia(:,end)>2);
arrhythmia(rows_arrythmic, end) = 2; 

arrhythmia(:, find(sum(abs(arrhythmia)) == 0)) = []; %remove null columns

class_id = arrhythmia(:,end);
y = arrhythmia(:,1: end-1);
[N,F]=size(y);
mean_y = mean(y);
std_y = std(y);
o=ones(N,1);% o is a column vector 
y=(y-o*mean_y)./(o*std_y);% y is normalized

%% Class 1 e 2

ind_sani = find(class_id==1)';
ind_malati = find(class_id==2)';
y1= y(ind_sani, :);
y2= y(ind_malati, :);
x1 = mean(y1);
x2 = mean(y2);

xmeans=[x1;x2]; % matrix with x1 and x2
eny=diag(y*y'); % |y(n)|^2
enx=diag(xmeans*xmeans'); % |x1|^2 and |x2|^2
rhoy=y*xmeans'; % matrix with the dot product between% each y(n) and each x
[U,V]=meshgrid(enx,eny); 
dist2=U+V-2*rhoy; %|y(n)|^2+|x(k)|^2-2y(n)x(k)= 
% =|y(n)-x(k)|^2


[dummy, previsione] = min(dist2');
previsione = previsione';

%%measurements

N1=sum(class_id==1);
N2=sum(class_id==2);
false_positive=sum((previsione==2)&(class_id==1))/N1;% 0.1633 
true_positive=sum((previsione==2)&(class_id==2))/N2;% 0.686 
false_negative=sum((previsione==1)&(class_id==2))/N2;% 0.314 
true_negative=sum((previsione==1)&(class_id==1))/N1;% 0.8367

%% before apply Bayes whiten the data, applying PCA

% perform PCA

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


z1=Z(ind_sani,:);
z2=Z(ind_malati,:); 
w1=mean(z1);
w2=mean(z2); 
wmeans=[w1;w2]; 
rhoz=Z*wmeans'; 
en1=diag(Z*Z');
en2=diag(wmeans*wmeans'); 
[Uy,Vy]=meshgrid(en2,en1); 
distz=Uy+Vy-2*rhoz; 
[a,decz]=min(distz,[],2);


false_positive_PCA=sum((decz==2)&(class_id==1))/N1;% 0.0612 
true_positive_PCA=sum((decz==2)&(class_id==2))/N2;% 0.8647
false_negative_PCA=sum((decz==1)&(class_id==2))/N2;% 0.1353
true_negative_PCA=sum((decz==1)&(class_id==1))/N1;% 0.9388

%% Bayesian Approach
%1
%To obtain the previous result, we removed the error related to 
%the assumption that the features are statistically independent; 
% but we are still making an error by assuming that the two classes have 
%the same probability, which is not

pis=zeros(1,2); 
pis(1)=N1/N;
pis(2)=N2/N;

dist2b=distz-2*o*log(pis);% from the square distance we remove 2*sig2*log(pi) 
[a,decb]=min(dist2b,[],2);
false_positive_b=sum((decb==2)&(class_id==1))/N1;% 0.0367
true_positive_b=sum((decb==2)&(class_id==2))/N2;% 0.8647
false_negative_b=sum((decb==1)&(class_id==2))/N2;% 0.1353 
true_negative_b=sum((decb==1)&(class_id==1))/N1;% 0.9633

%2
%NON FUNZIONA
% [N1,F1]=size(z1); 
% dd1=z1-ones(N1,1)*w1; 
% R1=dd1'*dd1/N1; 
% R1i=inv(R1);
%     
% [N2,F1]=size(z2); dd2=z2-ones(N2,1)*w1; R2=dd2'*dd2/N2; R2i=inv(R2);
% 
% G=zeros(N,2); 
% for n=1:N
% G(n,1)=(Z(n,:)-w1)*R1i*(Z(n,:)-w1)'+log(det(R1))-2*log(pis(1));
% G(n,2)=(Z(n,:)-w2)*R2i*(Z(n,:)-w2)'+log(det(R2))-2*log(pis(2)); 
% end
% [a,decbay]=min(G,[],2);
% false_positive_b2=sum((decbay==2)&(class_id==1))/N1;% 0 
% true_positive_b2=sum((decbay==2)&(class_id==2))/N2;% 0.9807 
% false_negative_b2=sum((decbay==1)&(class_id==2))/N2;% 0.0193 
% true_negative_b2=sum((decbay==1)&(class_id==1))/N1; % 1



%%%%% prova anche togliendo le feature/pazienti con diversi criteri (es.
%%%%% una colonna che ? tutta nulla tranne in un punto) => var = 0

