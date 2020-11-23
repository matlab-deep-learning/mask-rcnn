function [ T ] = generateRegressionTargets(groundTruthBoxes, proposals, assignment, positiveIdx, numClasses)

% assigned groundtruthBoxes - numProposalsx4
% proposals - numproposalsx4
% T - numProposals x numclasses

% Copyright 2020 The MathWorks, Inc.

% Compute regression targets for bbox  estimation 
% This function computes dx, dy, dw, dh from ground truth and proposals 


% Only compute targets for proposal assigned to boxes.

for i = 1:length(groundTruthBoxes)
    
    assignedGroundTruthBoxes{i} = zeros([size(assignment{i},1) 4],'like',groundTruthBoxes{1});
    assignedGroundTruthBoxes{i}(positiveIdx{i},:) = groundTruthBoxes{i}(assignment{i}(positiveIdx{i}),:);

    targets{i} = zeros(size(proposals{i},1),4,'like',proposals{i}); 
    targets{i}(positiveIdx{i},:) =...
        helper.BoundingBoxRegressionModel.generateRegressionTargets(...
                                assignedGroundTruthBoxes{i}(positiveIdx{i},:), proposals{i}(positiveIdx{i},:));

    T{i} = repmat(targets{i},1,numClasses); % -> [4*numClasses N], per class regression response.

end