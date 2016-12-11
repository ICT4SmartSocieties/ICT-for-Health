close all
clear all
% Lab 3 - Prepare the data

load('arrhythmia.mat');

for i = 1: length(arrhythmia)

    if(arrhythmia(i,end) > 1)
        arrhythmia(i,end) = 2;
    end
    
end

arrhythmiaCleaned=[];
deleted=[];
for i=1: size(arrhythmia,2)
    
    prova = arrhythmia(:,i) == 0;
    
    if( all(prova) )
        deleted = [deleted,i];
%         arrhythmiaCleaned = [arrhythmiaCleaned, arrhythmia(:,i)];
    else
%         deleted = [deleted,i];
        arrhythmiaCleaned = [arrhythmiaCleaned, arrhythmia(:,i)];
    end
  
end

save('arrhythmiaCleaned','arrhythmiaCleaned');



