function [clsResponse, T, W] = calculateClsRegressionResponses(trainingSamples, params)
%calculateClsRegressionResponses RPN image centric classification and regression responses.
%

% Copyright 2020 The MathWorks, Inc.

    % Samples contain one cell element of each of the struct fields.
    idx = 1;

    if isempty(trainingSamples.Positive) || ~any(trainingSamples.Positive)
        % There are usually many more negatives than positives, which is why do
        % not remove images that do not have any negatives.
        clsResponse = [];
        T           = [];
        W           = [];
        return;
    end

    if params.StandardizeRegressionTargets
        % Standardize regression targets
        meanAllTargets = params.BoxRegressionMean';
        stdAllTargets  = params.BoxRegressionStd';
        trainingSamples.PositiveBoxRegressionTargets{idx} = ...
                (trainingSamples.PositiveBoxRegressionTargets{idx} - meanAllTargets)./stdAllTargets;
    end

    samples    = trainingSamples.RegionProposals{idx};
    posSamples = samples(trainingSamples.Positive,:);

    % 1:1 ratio between positive and negatives.
    numPos = floor(params.RPNROIPerImage / 2);
    numPos = min(numPos, size(posSamples,1));

    % Create target matrix MxNxK (K == 2*num_anchors)
    ids          = trainingSamples.AnchorIndices{idx};
    anchorIDs    = trainingSamples.AnchorIDs{idx};
    posAnchorIDs = anchorIDs(trainingSamples.Positive);
    negAnchorIDs = anchorIDs(trainingSamples.Negative);

    posxy = ids(trainingSamples.Positive, :);
    negxy = ids(trainingSamples.Negative, :);

    featureMapSize = trainingSamples.FeatureMapSize{idx};
    clsResponse    = 3*ones([featureMapSize(1),featureMapSize(2),params.NumAnchors],'uint8');

    bb           = samples(trainingSamples.Positive,:);
    N            = size(bb,1);
    pid          = params.RandomSelector.randperm(N, min(N, numPos));
    posSamples   = posSamples(pid, :);
    posxy        = posxy(pid,:);
    posAnchorIDs = posAnchorIDs(pid);
    % pos
    for i = 1:numel(posAnchorIDs)
        x = posxy(i,1);
        y = posxy(i,2);
        k = posAnchorIDs(i);

        clsResponse(y, x, k) = 1; % for foreground class
    end

    bb = samples(trainingSamples.Negative,:);
    N  = size(bb,1);

    if params.MiniBatchPadWithNegatives
        numNeg = min(N, params.RPNROIPerImage - numPos);
    else
        % honor foreground fraction
        numNeg = min(N, floor(numPos/params.PercentageOfPositiveSamples - numPos));
    end

    id           = params.RandomSelector.randperm(N, min(N, numNeg));
    negxy        = negxy(id,:);
    negAnchorIDs = negAnchorIDs(id);

    % neg
    for i = 1:numel(negAnchorIDs)
        x = negxy(i,1);
        y = negxy(i,2);
        k = negAnchorIDs(i);

        clsResponse(y, x, k) = 2; % for background class
    end

    % reshape cls response into [H*W 2*NumAnchors]
    clsResponse = reshape(clsResponse, featureMapSize(1)*featureMapSize(2), []);
    negSamples  = bb(id,:);

    % training rois
    roi = [posSamples; negSamples];

    % REG Response
    posTargets = trainingSamples.PositiveBoxRegressionTargets{idx};
    posTargets = posTargets(:, pid);

    % Regression layer output is [M N 4*NumAnchors], where the 4
    % box coordinates are consequitive elements. W is a weight
    % matrix that indicates which sample should contribute to the
    % loss. W(y,x,k) is 1 if T(y,x,k) should be used.
    W = zeros(featureMapSize(1), featureMapSize(2), params.NumAnchors * 4, 'single');
    T = zeros(featureMapSize(1), featureMapSize(2), params.NumAnchors * 4, 'single');

    % Define instance weight
    if strcmp(params.BBoxRegressionNormalization,'batch')
        w = 1/size(roi,1);
    else
        % valid - treat each positive sample as an observation.
        w = 1/max(1,4*size(posSamples,1));
    end

    % Put positive targets into appropriate location by anchor ID.
    % Only the positive anchors need to be included because they
    % are the only one that get regressed.
    for i = 1:numel(posAnchorIDs)
        x     = posxy(i,1);
        y     = posxy(i,2);
        k     = posAnchorIDs(i);
        start = 4*(k-1) + 1;
        stop  = start+4-1;

        W(y,x,start:stop) = w;
        T(y,x,start:stop) = posTargets(:,i);
    end

    % Convert classification response to categorical.
    clsResponse = params.CategoricalLookup(clsResponse);
end
