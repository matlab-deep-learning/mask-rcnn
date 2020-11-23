function [gradients, totalLoss, state] = networkGradients(X, gTruthBoxes, gTruthLabels, gTruthMasks, dlnet, params)
% networkGradients - Gradient function to train MaskRCNN using a custom
% training loop.
 
% Copyright 2020 The MathWorks, Inc.


RPNRegDeltas = {'rpnConv1x1BoxDeltas'};regionProposal = {'rpl'};
 
outputNodes = [ RPNRegDeltas, regionProposal, dlnet.OutputNames(:)'];
 
% For training, the first step is to run a forward pass on the MaskRCNN
% network.
% We need the following outputs from the network to compute losses -
% YRPNReg - RPN regression output  [1x1x4xN] {[deltaX deltaY deltaW deltaH]}
% YRPNClass - RPN classification output [1x1x2xN]
% YRCNNReg - RCNN regression output  [1x1x4*(numClasses)xN] {[deltaX deltaY deltaW deltaH]}
% YRCNNClass - RCNN classification output [1x1x(numClasses+1)xN]
% YMask     - RCNN Mask segmentation output [hxwxnumClassesxN]
% proposals - The regionproposals from the RPN. [5xN]
[YRPNRegDeltas, proposal, YRCNNClass, YRCNNReg, YRPNClass, YMask, state] = forward(...
                                            dlnet, X, 'Outputs', outputNodes);

% If there are no proposals don't learn anything
if(isempty(proposal))
   totalLoss = dlarray([]);
   gradients= dlarray([]);
   disp("Empty proposals");
   return;
end

% Proposals are 5XNumProposals (Due to batch restrictions from custom RPL layer)
proposals = gather(extractdata(proposal));

% Convert proposals to numProposals x 5 (as expected by the rest of post processing code)
proposals =proposals';

proposals(:,1:4) = helper.boxUtils.x1y1x2y2ToXYWH(proposals(:,1:4));

numImagesInBatch = size(gTruthBoxes,1);
%Convert numProposalsx5 Proposals to numImagesInBatchx1 (Group by image index)
proposals = helper.groupProposalsByImageIndex(proposals, numImagesInBatch);


% Generate RCNN response targets
%--------------------------------

% Step 1: Match ground truth boxes to proposals                                       
[assignment, positiveIndex, negativeIndex] = helper.bboxMatchAndAssign(...
                                                        proposals, gTruthBoxes,...
                                                        params.PositiveOverlapRange, params.NegativeOverlapRange,...
                                                        false);

                                                    
% Step 2: Calcuate regression targets as (dx, dy, log(dw), log(dh))
regressionTargets = helper.generateRegressionTargets(gTruthBoxes, proposals,...
                                                        assignment, positiveIndex,...
                                                        params.NumClasses);
 
classNames = categories(gTruthLabels{1});

% Step 3: Assign groundtrutrh labels to proposals
classificationTargets = helper.generateClassificationTargets (gTruthLabels, assignment,...
                                             positiveIndex, negativeIndex,...
                                             classNames, params.BackgroundClass);
                                         
% Step 4: Calculate instance weights
instanceWeightsReg = helper.regressionResponseInstanceWeights (classificationTargets, params.BackgroundClass);
 
% Step 5: Generate mask targets
 
% Crop and resize the instances based on proposal bboxes and network output size
maskOutputSize = params.MaskOutputSize;
croppedMasks = helper.cropandResizeMasks (gTruthMasks, gTruthBoxes, maskOutputSize);
 
% Generate mask targets
maskTargets = helper.generateMaskTargets(croppedMasks, assignment, classificationTargets, params);

% Stage 2 (RCNN) Loss
% --------------------

% *Classification loss*
classificationTargets = cat(1, classificationTargets{:})';
% onehotencode labels
classificationTargets = onehotencode(classificationTargets,1);
classificationTargets(isnan(classificationTargets)) = 0;

LossRCNNClass = loss.CrossEntropy(YRCNNClass, classificationTargets);
 
% *Weighted regression loss*
regressionTargets = cat(1,regressionTargets{:});
instanceWeightsReg = cat(1, instanceWeightsReg{:});

LossRCNNReg = loss.smoothL1(YRCNNReg, single(regressionTargets'), single(instanceWeightsReg'));
 
% Mask Loss (Weighted cross entropy)
maskTargets= cat(4,maskTargets{:});
positiveIndex = cat(1,positiveIndex{:});
LossRCNNMask = loss.SpatialCrossEntropy(YMask, single(maskTargets), positiveIndex);
 
% Total Stage 2 loss
 LossRCNN = LossRCNNReg + LossRCNNClass + LossRCNNMask;

 
% Generate RCNN response targets
%--------------------------------
featureSize = size(YRPNRegDeltas);
imageSize = params.ImageSize;
[RPNRegressionTargets, RPNRegWeights, assignedLabelsRPN] = helper.rpnRegressionResponse(featureSize, gTruthBoxes, imageSize, params);

RPNClassificationTargets = onehotencode(assignedLabelsRPN, 3);
RPNClassificationTargets(isnan(RPNClassificationTargets)) = 0;


% Stage 1 (RPN) Loss
% --------------------
   
LossRPNClass = loss.CrossEntropy(YRPNClass, RPNClassificationTargets);
 
LossRPNReg = loss.smoothL1(YRPNRegDeltas, RPNRegressionTargets, RPNRegWeights);
 
LossRPN = LossRPNClass + LossRPNReg;


% Total Loss
%------------
totalLoss = LossRCNN + LossRPN;
 
gradients = dlgradient(totalLoss, dlnet.Learnables);
 
end