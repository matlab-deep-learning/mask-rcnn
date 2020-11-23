function [ assignments, positiveIndex, negativeIndex] = bboxMatchAndAssign(proposals, gTruthBoxes,...
                                                         posOverlap, negOverlap,...
                                                         assignAllGTruthBoxes)
% bboxMatchAndAssign assigns a box to a ground truth box if the
% IoU is between the range specified in posOverlap. If the IoU is between
% the range specified in negOverlap, the box is defined as a negative box.
%
% Inputs:
%   boxes: A batchSize-by-1 cell array of boxes (M1-by-4) stored as 
%          [x y width height] bounding boxes.
%
%   groundTruthBoxes: A batchSize-by-1 cell array of ground truth boxes (M2-by-4)
%                     stored as [x y width height] bounding boxes.
%
%   posOverlap: A two-element vector that specifies a range of
%               bounding box overlap ratios between 0 and 1.
%               Region proposals that overlap with ground truth
%               bounding boxes within the specified range are
%               used as positive training samples.
%
%   negOverlap: A two-element vector that specifies a range of
%               bounding box overlap ratios between 0 and 1.
%               Region proposals that overlap with ground truth
%               bounding boxes within the specified range are
%               used as negative training samples.
%
%   assignAllGroundTruthBoxes: A logical scalar. When true, all
%                              ground truth boxes are assigned
%                              to one of the boxes regardless
%                              of the positive overlap ratio.
%                              This is set to true in RPN and
%                              SSD.
%
% Outputs:
%   assignments:   Indices of ground truth boxes assigned to
%                  each box. A batchSize-by-1 cell array holding vector of 
%                  length size(boxes{i},1).
%                  If a box is not assigned to a ground truth
%                  box, the assignment index is 0.
%
%   positiveIndex: A batchSize-by-1 cell array holding vector of logical 
%                  indices of length size(boxes{i},1). A box is "positive" 
%                  if the overlap threshold with a ground truth box is
%                  within the range specified in input
%                  posOverlap.
%
%   negativeIndex: A batchSize-by-1 cell array holding vectors of logical 
%                  indices of length size(boxes{i},1). A box is "negative" 
%                  if the overlap threshold with a ground truth box is
%                  within the range specified in input
%                  negOverlap.
     
% Copyright 2020 The MathWorks, Inc.    
     numImagesInBatch = size(gTruthBoxes,1); 
     
     % Loop over batches
     for i = 1:numImagesInBatch
         
         proposalsInBatch = proposals{i};
         gTruthBoxesInBatch = gTruthBoxes{i};
         
         [assignments{i},positiveIndex{i}, negativeIndex{i}]=...
                            helper.boxAssignmentUtils.assignBoxesToGroundTruthBoxes(...
                                proposalsInBatch(:,1:4), gTruthBoxesInBatch,...
                                posOverlap, negOverlap,...
                                assignAllGTruthBoxes);
         
     
     
     end
end