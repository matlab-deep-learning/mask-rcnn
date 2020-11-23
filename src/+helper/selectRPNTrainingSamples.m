function samples = selectRPNTrainingSamples(imageSize, featureMapSize, params, varargin)

% Copyright 2020 The MathWorks, Inc.

if isfield(params, 'GroundTruthMby4') && params.GroundTruthMby4
    % Ground truth is already vertcat'ed into one M-by-4 matrix.
    groundTruthBoxes = varargin{1};
else
    % varargin is 1 row of the ground truth table.

    % cat all multi-class bounding boxes into one M-by-4 matrix.
    groundTruthBoxes = vertcat(varargin{2:numel(varargin)});
end

% Scale factor from feature to image space.
scaleFactor = 1./params.ScaleFactor;

% generate box candidates
[regionProposals, anchorLocInFeatureMap] = generateAnchorBoxesInImage(...
    imageSize, featureMapSize, params.AnchorBoxes, scaleFactor);

% create anchor Ids for each anchor box. these are required to
% assign each target to the correct box regressor.
numAnchors = cellfun(@(x)size(x,1), regionProposals);
anchorIDs = repelem(1:numel(regionProposals), numAnchors);

% convert from k cells to M-by-2 format.
regionProposals = (vertcat(regionProposals{:}));
anchorIndices = (vertcat(anchorLocInFeatureMap{:}));

matchAllGroundTruth = true; % every ground truth box should be assigned to a box.
[targets, positiveIndex, negativeIndex] = ...
    vision.maskrcnn.utils.BoundingBoxRegressionModel.generateRegressionTargetsFromProposals(...
    regionProposals, groundTruthBoxes, params.PositiveOverlapRange, params.NegativeOverlapRange, matchAllGroundTruth);

% foregound labels are located @ 1:k. bg labels are @ k+1:2k.
labels = anchorIDs;
labels(negativeIndex) = labels(negativeIndex) + params.NumAnchors;
labels = categorical(labels, 1:(2*params.NumAnchors));

% Sub-sample negative samples to avoid using too much memory.
numPos = sum(positiveIndex);
negIdx = find(negativeIndex);
numNeg = numel(negIdx);
nidx   = params.RandomSelector.randperm(numNeg, min(numNeg, 5000));

% Pack data as int32 to save memory.
regionProposals = int32([regionProposals(positiveIndex, :); regionProposals(nidx, :)]);
anchorIDs       = {int32([anchorIDs(positiveIndex) anchorIDs(nidx)])};
anchorIndices   = {int32([anchorIndices(positiveIndex,:); anchorIndices(nidx,:)])};

labels = {[labels(positiveIndex) labels(nidx)]};

nr = size(regionProposals,1);
positiveIndex = false(nr,1);
negativeIndex = false(nr,1);

positiveIndex(1:numPos) = true;
negativeIndex(numPos+1:end) = true;

% return the region proposals, which may have been augmented
% with the ground truth data.
regionProposals = {regionProposals};

featureMapSize = {featureMapSize};

samples = struct('Positive', {positiveIndex}, ...
    'Negative',{negativeIndex}, ...
    'Labels', {labels}, ...
    'RegionProposals', {regionProposals}, ...
    'PositiveBoxRegressionTargets', {targets}, ...
    'AnchorIDs',{anchorIDs}, ...
    'AnchorIndices',{anchorIndices}, ...
    'FeatureMapSize', {featureMapSize});

end
