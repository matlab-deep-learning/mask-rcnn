function params = createMaskRCNNConfig(imageSize, numClasses, classNames)
% createNetworkConfiguration creates the maskRCNN training and detection
% configuration parameters

% Copyright 2020 The MathWorks, Inc.


    % Network parameters
    params.ImageSize = imageSize;
    params.NumClasses = numClasses;
    params.ClassNames = classNames;
    params.BackgroundClass = 'background';
    params.ROIAlignOutputSize = [14 14]; % ROIAlign outputSize
    params.MaskOutputSize = [14 14]; 
    params.ScaleFactor = [0.0625 0.0625]; % Feature size to image size ratio
    params.ClassAgnosticMasks = true;   
        
    % Target generation params 
    params.PositiveOverlapRange = [0.6 1.0];
    params.NegativeOverlapRange = [0.1 0.6];
       
    % Region Proposal network params
    params.AnchorBoxes = [[32 16];
                          [64 32];
                          [128 64];
                          [256 128];
                          [512 256];
                          [32 32];
                          [64 64];
                          [128 128];
                          [256 256];
                          [512 512];
                          [16 32];
                          [32 64];
                          [64 128];
                          [128 256];
                          [256 512]];
    params.NumAnchors = size(params.AnchorBoxes,1);
    params.NumRegionsToSample = 200;
    % NMS threshold
    params.OverlapThreshold = 0.7;
    params.MinScore = 0;
    params.NumStrongestRegionsBeforeProposalNMS = 3000;
    params.NumStrongestRegions = 1000;
    params.BoxFilterFcn = @(a,b,c,d)fasterRCNNObjectDetector.filterBBoxesBySize(a,b,c,d);
    params.RPNClassNames = {'Foreground', 'Background'};
    params.RPNBoxStd   = [1 1 1 1];
    params.RPNBoxMean  = [0 0 0 0];

    params.RandomSelector = vision.internal.rcnn.RandomSelector();
    params.StandardizeRegressionTargets = false;
    params.MiniBatchPadWithNegatives = true;
    params.ProposalsOutsideImage = 'clip';
    params.BBoxRegressionNormalization = 'valid';
    params.RPNROIPerImage = params.NumRegionsToSample;
    params.CategoricalLookup = reshape(categorical([1 2 3],[1 2],params.RPNClassNames),[],1);
       
    % Detection params
    params.DetectionsOnBorder = 'clip';
    params.Threshold = 0.5;
    params.SelectStrongest = true;
    params.MinSize     = [1 1];
    params.MaxSize     = [inf inf];
