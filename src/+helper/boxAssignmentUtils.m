classdef boxAssignmentUtils
    % This class provides utilities for assigning boxes to ground truth
    % boxes. This is used for training anchor/box prior based CNN object
    % detectors such as RPN, Fast R-CNN, Faster R-CNN, MaskRCNN, SSD, YOLO v2, etc.    
    
    % Copyright 2020 The MathWorks, Inc.
    
    methods(Static)
        
        %------------------------------------------------------------------
        function [assignments, positiveIndex, negativeIndex] = assignBoxesToGroundTruthBoxes(...
                boxes, groundTruthBoxes, posOverlap, negOverlap, assignAllGroundTruthBoxes)
            % assignBoxesToGroundTruthBoxes assigns a box to a ground truth box if the
            % IoU is between the range specified in posOverlap. If the IoU is between
            % the range specified in negOverlap, the box is defined as a negative box.
            %
            % Inputs:
            %   boxes: An array of boxes (M1-by-4) stored as [x y width
            %          height] bounding boxes.               
            %
            %   groundTruthBoxes: An array of ground truth boxes (M2-by-4)
            %                     stored as [x y width height] bounding
            %                     boxes.
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
            %                  each box. A vector of length size(boxes,1).
            %                  If a box is not assigned to a ground truth
            %                  box, the assignment index is 0.
            %
            %   positiveIndex: A vector of logical indices of length
            %                  size(boxes,1). A box is "positive" if the
            %                  overlap threshold with a ground truth box is
            %                  within the range specified in input
            %                  posOverlap.
            %
            %   negativeIndex: A vector of logical indices of length
            %                  size(boxes,1). A box is "negative" if the
            %                  overlap threshold with a ground truth box is
            %                  within the range specified in input
            %                  negOverlap.
            
            
            % Compute the Intersection-over-Union (IoU) metric between the
            % ground truth boxes and the region proposal boxes.
            iou = helper.boxAssignmentUtils.bboxOverlapIoU(groundTruthBoxes, boxes);
            
            % Assign based on IoU metric.
            [assignments, positiveIndex, negativeIndex] = ...
                helper.boxAssignmentUtils.assignBoxes(iou, posOverlap, negOverlap, assignAllGroundTruthBoxes);
            
        end
        
        %------------------------------------------------------------------
        function [assignments, positiveIndex, negativeIndex] = assignBoxes(...
                iou, posOverlap, negOverlap, assignAllGroundTruthBoxes)
            % Input iou has ground truth boxes along first dimension and
            % boxes to assign along second dimension.
            %
            % Each box is assigned to exactly one ground truth.
            % Each ground truth can be assigned to more than 1 box.
            
            assert(numel(posOverlap)==2);
            assert(numel(negOverlap)==2);
            
            % For each box, find best matching GT box (one with the largest
            % IoU score).
            [v,idx] = max(iou,[],1);
            
            % Assign each anchor a positive label if overlap threshold is
            % within specified range.
            lower = posOverlap(1);
            upper = posOverlap(2);
            positiveIndex =  reshape(v >= lower & v <= upper,[],1);
            
            % Ensure each ground truth box is assigned to at least one box.
            % This type of matching is used by RPN and SSD.
            if assignAllGroundTruthBoxes && ~isempty(iou)
                
                % Create a mask to set all assigned box IoU to 1. 
                j = 1:size(iou,2);
                ind = sub2ind(size(iou),idx(positiveIndex),j(positiveIndex));
                mask = false(size(iou));
                mask(ind) = true;                
                iou(mask) = 1;
                
                % Unmask box columns that haven't been assigned to a ground
                % truth box.
                mask(:,~positiveIndex) = true;                                                
                iou = mask.*iou;
                
                % For each GT box, find box with largest overlap that
                % hasn't already been assigned to a ground truth. A box can
                % only be assigned to a single ground truth box. 
                [vg,~] = max(iou,[],2);
                
                % ignore gTruth that has zero IoU.
                vg(vg==0) = -inf; 
                
                % Assign these boxes as positive to ensure each ground
                % truth box is included. There may be multiple anchors with
                % equal max overlap values. Find these and assign them all
                % positive labels.
                [idx1, allMatches] = find(mask.*(iou==vg));
                
                % Update original set of indices.
                idx(allMatches) = idx1;
                positiveIndex(allMatches) = true;               
            end
            
            % Generate a list of the ground truth box assigned to each
            % positive box.
            assignments = zeros(size(iou,2),1,'single');
            assignments(positiveIndex) = idx(positiveIndex);
            
            % Select regions to use as negative training samples
            lower = negOverlap(1);
            upper = negOverlap(2);
            negativeIndex =  reshape(v >= lower & v < upper,[],1);
            
            % Boxes marked as positives should not be negatives.
            negativeIndex(positiveIndex) = false;
        end
        
        %------------------------------------------------------------------
        function iou = bboxOverlapIoU(groundTruthBoxes,boxes)
            % Return IoU given ground truth boxes and boxes (anchors/box
            % priors or region proposals).
            %
            % Output iou is a M-by-N matrix, where M is
            % size(groundTruthBoxes,1) and N is size(boxes,1).            
            if isempty(groundTruthBoxes)
                iou = zeros(0,size(boxes,1));
            elseif isempty(boxes)
                iou = zeros(size(groundTruthBoxes,1),0);
            else
                iou = bboxOverlapRatio(groundTruthBoxes(:,1:4), boxes, 'union');
            end
            
        end
        
        %------------------------------------------------------------------
        function labels = boxLabelsFromAssignment(...
                assignments, groundTruthBoxLabels, ...
                positiveIndex, negativeIndex, ...
                classNames, backgroundClassName)            
            % Return categorical vector of class names assigned to each box.
            
            % Preallocate categorical output.
            labels = repmat({''}, size(assignments,1), 1);
            labels = categorical(labels,[reshape(classNames,[],1); backgroundClassName]);

            labels(positiveIndex,:) = groundTruthBoxLabels(assignments(positiveIndex),:);
            labels(negativeIndex,:) = {backgroundClassName};  
            
        end
    end
end

