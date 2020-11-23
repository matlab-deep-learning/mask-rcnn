function [ outputMasks ] = cropandResizeMasks (gTruthMasks, gTruthBoxes, outputSize)

% gTruthMasks - HxWxnumObjects
% gTruthBoxes - numObjectsx4
% outputSize - [h' w']
% outputMasks - h'x w' x numObjects

% Copyright 2020 The MathWorks, Inc.

% Loop through the gtruth boxes and crop corresponding mask instance

for i = 1:length(gTruthMasks)

    numObjects = size(gTruthBoxes{i},1);
    outputMasks{i} = zeros([outputSize numObjects], 'single');

    for j=1:numObjects

        croppedMask = imcrop(gTruthMasks{i}(:,:,j), gTruthBoxes{i}(j,:));

        outputMasks{i}(:,:,j) = imresize(croppedMask, outputSize, 'nearest');

    end
end
    
    
                            