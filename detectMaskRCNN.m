function [bboxes, scores, labels, finalMasks] = detectMaskRCNN(dlnet, maskSubnet, image, params, executionEnvironment)

% detectMaskRCNN runs prediction on a trained maskrcnn network
%
% Inputs:
% dlnet      - Pretrained MaskRCNN dlnetwork
% maskSubnet - Pretrained mask branch of maskRCNN dlnetwork
% image      - RGB image to run prediction on. (H x W x 3)
% params     - MaskRCNN network configuration object Created using
%               helper.createNetworkConfiguration
%
% Outputs:
% bboxes     - Final bounding box detections ([x y w h]) formatted as
%              NumDetections x 4.
% scores     - NumDetections x 1 classification scores.
% labels     - NumDetections x 1 categorical class labels.
% finalMasks - Binary object Masks detections formatted as 
%              H x W x NumDetections

% Copyright 2020 The MathWorks, Inc.


% Prepare input image for prediction
if(executionEnvironment == "gpu")
    image = gpuArray(image);
end
% Cast the image to a dlarray
X = dlarray(single(image),'SSCB');
imageSize = size(image);


%%% Detector prediction

% Outputs for object detection
featureMapNode = 'res4b22_relu';
outputNodes = {'rpl', 'rcnnSoftmax', 'fcBoxDeltas'};
outputNodes = [outputNodes featureMapNode];

% Run prediction on the inputs
[bboxes, YRCNNClass, YRCNNReg, featureMap] = predict(...
                                        dlnet, X, 'Outputs', outputNodes);
                                    
% Extract data from the output dlarrays
bboxes = extractdata(bboxes)';
YRCNNClass = extractdata(YRCNNClass);
YRCNNReg = extractdata(YRCNNReg);

% Network outputs proposals in [x1 y1 x2 y2] format in the image
% space. Convert to [x y w h] for further processing.
bboxes = vision.internal.cnn.boxUtils.x1y1x2y2ToXYWH(bboxes);

% Classification data processing
% Calculate scores
allScores = squeeze(YRCNNClass);
% remove scores associated with background
bgIndex = strcmp(string(params.ClassNames),params.BackgroundClass);
allScores(bgIndex,:) = [];
scores = reshape(allScores',[],1);

numClasses = numel(params.ClassNames)-1; % exclude background

% replicate each proposal box for each class.
% The boxes are replicated as [b1 b1 b1 b1 b2 b2 b2 b2 ....]
bboxes = repmat(bboxes(:, 1:4), numClasses, 1);
numObservations = size(YRCNNReg,2);

 % replicate labels 
 % Group the labels as [label1 label1 ... numObservation times, label2
 % label2.... numObservation times ...]'
classNames = categorical(params.ClassNames, params.ClassNames);
% Remove background
classNames(classNames==params.BackgroundClass) = [];
classNames = removecats(classNames, params.BackgroundClass);

ind = (1:numel(classNames))';
labels = classNames(repelem(ind,numObservations,1));
labels = labels';


% Regression data processing
% YRCNNreg is  [numClasses*4 numObs]. reshape
% to 4-by-(numClasses * numObs), where each column is
% the target for a specific classes. Data is arranged
% as follows:
%
%   [c1 c2 c1 c2 c1 c2], where rows stride by
%   numClasses goes to next observation.
reg = reshape(YRCNNReg, 4, numClasses, numObservations);
reg = permute(reg, [1 3 2]);
reg = reshape(reg, 4, [])';


% Apply regression
bboxes = helper.applyRegression(bboxes, reg, params.MinSize, params.MaxSize);

% filter invalid predictions
[bboxes, scores, labels] = helper.filterBoxesAfterRegression(bboxes,scores,labels, imageSize);

% Filter boxes clip boxes on the edge
bboxes = vision.internal.detector.clipBBox(bboxes,imageSize);

% Remove boxes that are too small after clipping.
tooSmall = any(bboxes(:,3:4) < params.MinSize,2);
bboxes(tooSmall,:) = [];
scores(tooSmall,:) = [];
labels(tooSmall,:) = [];
 

% Filter scores lesser than the Threshold
keep = scores >= params.Threshold;
bboxes = bboxes(keep,1:4);
scores = scores(keep,:);
labels = labels(keep,:);

% Perform NMS
if params.SelectStrongest
    [bboxes, scores, labels] = selectStrongestBboxMulticlass(bboxes,scores,labels,...
        'RatioType', 'Min', 'OverlapThreshold', 0.7);
end

%%% Mask segmentation

% Prepare final bboc detections for mask prediction
bboxesX1Y1 = vision.internal.cnn.boxUtils.xywhToX1Y1X2Y2(bboxes);
roiIn = dlarray([bboxesX1Y1 ones(size(bboxesX1Y1,1),1)]', "SSCB");

mask = predict(maskSubnet, roiIn, featureMap);

bboxes = gather(bboxes);
mask = gather(squeeze(extractdata(mask)));

finalMasks = false([imageSize(1) imageSize(2) size(bboxes,1)]);

% Resize and insert masks
for i = 1:size(bboxes,1)
    m = imresize(mask(:,:,i), [bboxes(i,4) bboxes(i,3)],'cubic') > 0.5 ;
    finalMasks(bboxes(i,2):bboxes(i,2)+bboxes(i,4)-1, ...
                bboxes(i,1):bboxes(i,1)+bboxes(i,3)-1, i) = m;
end


end



