function [assignedMasks] = generateMaskTargets (croppedMasks, assignments, assignedLabels, params)

% croppedMasks - h'xw'xnumObjects
% assignments - numProposalsx1
% assignedMasks - h'x w'x numClasses x numProposals

% Copyright 2020 The MathWorks, Inc.

for i = 1:length(croppedMasks)
    
    if(params.ClassAgnosticMasks)
        numMaskClasses = 1;
    else
        numMaskClasses = params.NumClasses;
    end
    
    labelIds = uint8(assignedLabels{i});
    numProposals = size(assignments{i},1);
    assignedMasks{i} = zeros([size(croppedMasks{i},1) size(croppedMasks{i},2) numMaskClasses numProposals]);

    assignedIdxs = find(assignments{i}~=0);
    
    for j = 1:length(assignedIdxs)
        
        if(params.ClassAgnosticMasks)
            classIdx = 1;
        else
            classIdx = labelIds(assignedIdxs(j));
        end
        
        assignedMasks{i}(:,:,classIdx,assignedIdxs(j)) = croppedMasks{i}(:,:, assignments{i}(assignedIdxs(j)));
    end

end
