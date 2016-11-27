%lab01 - patient with parkinson %
clear all
close all 
clc

load('updrs.mat');
originalMatrix = parkinsonsupdrs;
timeColumn = floor(originalMatrix(:,4));
originalMatrix(:,4) = timeColumn;

%elimante times<0 and times> 180
lineToKeep = originalMatrix(:,4) >=0 & originalMatrix(:,4) <=180;
cleanedMatrix = originalMatrix(lineToKeep,:);
cleanedMatrix(1,:)=[];
%sort per patientID and measurment times
groupedMatrix = sortrows(sortrows(cleanedMatrix,4),1);

[row,col] = size(groupedMatrix);
line =1;
addArray=zeros(1,col);
newMatrix=zeros(1,col);

%toSumLines = groupedMatrix(:,1) == p & groupedMatrix(:,4) == tc


%averaging the same (pID,time) rows 
while line < row %for all line
    p=groupedMatrix(line, 1);
    t=groupedMatrix(line, 4);
    
    %functions that returns the dimension of same (pID,time) block
    s = innerBlock(p,line,t,groupedMatrix);
    
    for inner=0:(s-1)
        addArray = addArray + groupedMatrix(line+inner,:);
    end
    newMatrix = vertcat(newMatrix,addArray./(s));
    
    addArray=zeros(1,22);
    line = line + s;   
end
newMatrix(1,:)=[];

%%%%%%%%%%%%%%
% regression %
%%%%%%%%%%%%%%

%devides the patients in two sets
data_train = newMatrix(newMatrix(:,1) <=36,:);
data_test = newMatrix(newMatrix(:,1) >36,:);


%normalizzation
m_data_train=mean(data_train(:,5:22),1); %mean for each column
v_data_train=var(data_train(:,5:22)); %variance normalized for #elements -1
m_data_test=mean(data_test(:,5:22),1); %mean for each column
v_data_test=var(data_test(:,5:22)); %variance normalized for #elements -1
 
%normalizing on the column 7-22 by using the  m/v vectors
data_train_norm1 = data_train(:,1:4);
data_train(:,1:4) = [];
m = ones(size(data_train,1),1);
matrix_means = m*m_data_train;
matrix_vars = m*v_data_train;
data_train_norm2 = (data_train - matrix_means) ./sqrt(matrix_vars);
data_train_norm = [data_train_norm1 data_train_norm2];

data_test_norm1 = data_test(:,1:4);
data_test(:,1:4) = [];
m = ones(size(data_test,1),1);
matrix_means = m*m_data_test;
matrix_vars = m*v_data_test;
data_test_norm2 = (data_test - matrix_means) ./sqrt(matrix_vars);
data_test_norm = [data_test_norm1 data_test_norm2];

% MSE prediction
F5 = 5;
F7 = 7;
MSEprediction(data_train_norm,data_test_norm,F5);
MSEprediction(data_train_norm,data_test_norm,F7);

%gradient algorithm predicition
gradientAlgorithm(data_train_norm, data_test_norm, F5);
gradientAlgorithm(data_train_norm, data_test_norm, F7);

%steepest descent
steepestDescent(data_train_norm,data_test_norm,F5);
steepestDescent(data_train_norm,data_test_norm,F7);
