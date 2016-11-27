function newMatrix=loadData()
    load('updrs.mat');
    originalMatrix = parkinsonsupdrs;
    originalMatrix(:,4) = abs(floor(originalMatrix(:,4)));
    groupedMatrix = sortrows(sortrows(originalMatrix,4),1);

    [row,col] = size(groupedMatrix);
    line =1;
    addArray=zeros(1,col);
    newMatrix=zeros(1,col);

    %averaging the same (pID,time) rows 
    while line < row %row %for all line
        p=groupedMatrix(line, 1);
        t=groupedMatrix(line, 4);

        %functions that returns the dimension of same (pID,time) block
        s = innerBlock(p,line,t,groupedMatrix);

        for inner=0:(s-1)
            addArray = addArray + groupedMatrix(line+inner,:);
        end
        %newMatrix = vertcat(newMatrix,addArray./(s));
        newMatrix = [newMatrix; addArray./s];

        addArray=zeros(1,22);
        line = line + s; 
    end
    newMatrix(1,:)=[];
end