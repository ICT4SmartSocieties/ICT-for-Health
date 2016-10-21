close all
clear all
clc

load('arrhythmia.mat');
rows_arrythmic = find(arrhythmia(:,end)>1);
arrhythmia(rows_arrythmic, end) = 2; 
j=1;
for i = 1:size(arrhythmia,2)
    if (unique(arrhythmia(:,i))==0)
       col_del(j) = i;
       j = j+1;
    end
end
col_del = sort(col_del, 'descend');
for i = col_del
    arrhythmia(:,i) = [];
end

class_id = arrhythmia(:,end);
y = arrhythmia(:,1: end-1);


y1= y(find(class_id<2), :);
y2= y(find(class_id>1), :);

x1 = mean(y1);
x2 = mean(y2);

 xmeans=[x1;x2];% matrix with x1 and x2
eny=diag(y*y');% |y(n)|^2
enx=diag(xmeans*xmeans');% |x1|^2 and |x2|^2
dotprod=y*xmeans';% matrix with the dot product between
% each y(n) and each x
[U,V]=meshgrid(enx,eny); dist2=U+V-2*dotprod;%|y(n)|^2+|x(k)|^2-2y(n)x(k)= %=|y(n)-x(k)|^2