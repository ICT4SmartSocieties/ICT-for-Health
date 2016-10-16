close all
clear all
%Lab 1 - ICT HEALTH - Prepare the data

load('updrs.mat');

Npacients = 42;

parkinsonsupdrs(:,4) = abs(floor(parkinsonsupdrs(:,4)));

finalMatrix = [];
for pacient = 1: Npacients
    
    rowPacient = parkinsonsupdrs(:, 1) == pacient;
    matrixPacient = parkinsonsupdrs(rowPacient,:);
    
    count = 0;
    for day = 1:max(parkinsonsupdrs(:,4))+1
       indx = matrixPacient(:,4) == day-1;
       if (any(indx(:) == 1))
           count = count+1;
           meanMatrixPacient(count,:) = mean(matrixPacient(indx,:));
       end
    end
    
    finalMatrix = [finalMatrix; meanMatrixPacient];
    meanMatrixPacient=[];

end

save('finalMatrix','finalMatrix');