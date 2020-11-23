function [Treg, Wreg, assignedLabels] = rpnRegressionResponse( featureSize, gTruthBoxesInBatch, imageSize, params)
    
% Copyright 2020 The MathWorks, Inc.
    proposalParams = params;

    numObservations = numel(gTruthBoxesInBatch);
    data            = cell(numObservations, 3);
    for ii = 1:numObservations
        samples     = helper.rpnTrainingSamples(imageSize, featureSize, proposalParams, gTruthBoxesInBatch{ii});
        [data{ii, 1}, data{ii, 2}, data{ii, 3}] = vision.internal.cnn.rpn.calculateClsRegressionResponses(samples, params);
    end

    % Categorical response labels will be a 
    % numObersvations size of each H*W-by-3*numAnchors.
    % Categorical responses need to be a 4-D array.
    assignedLabels = cat(4, data{:,1});

    % Regression responses are numObservations size of each
    % H-by-W-by-4*numAnchors array.
    Treg = cat(4, data{:,2});
    Wreg = cat(4, data{:,3});
end