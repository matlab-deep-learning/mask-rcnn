function data = preprocessData(data, targetSize)
% Resize the image and scale the corresponding bounding boxes.

% Copyright 2020 The MathWorks, Inc.

im = data{1};
bboxes = data{2};
labels = data{3};
masks = data{4};


imgSize = size(im);

% Resize the min dimension to targetSize and resize the other dim to
% maintain aspect ratio. then crop the image to targetSize

% Resize images, masks and bboxes
[~, minDim] = min(imgSize(1:2));

resizeSize = [NaN NaN];
resizeSize(minDim) = targetSize(minDim);

im = imresize(im,resizeSize);
masks = imresize(masks,resizeSize);

resizeScale = targetSize(minDim)/imgSize(minDim);

bboxes = bboxresize(round(bboxes),resizeScale);

% Crop to target size
cropWindow = randomCropWindow2d(size(im), targetSize(1:2));

[bboxes, indices] = bboxcrop(bboxes, cropWindow, 'OverlapThreshold', 0.7);

im = imcrop(im, cropWindow);

[r,c] = deal(cropWindow.YLimits(1):cropWindow.YLimits(2),cropWindow.XLimits(1):cropWindow.XLimits(2));
masks = masks(r,c,indices);

labels = labels(indices);

% im_out = zeros(targetSize, 'like', im);
% masks_out = false([targetSize(1:2) size(masks,3)]);
% 
% 
% imgSize = size(im);
% 
% % Resize the max dimension to targetSize and resie the other dim to
% % maintain aspect ratio
% [~, maxDim] = max(imgSize);
% 
% resizeSize = [NaN NaN];
% resizeSize(maxDim) = targetSize(maxDim);
% 
% im = imresize(im,resizeSize);
% masks = imresize(masks,resizeSize);
% 
% % Pad the rest of the image to maintain targetSize
% rescaledImSize = size(im);
% 
% im_out(1:rescaledImSize(1), 1:rescaledImSize(2), :) = im;
% masks_out(1:rescaledImSize(1), 1:rescaledImSize(2), :) = masks;
% 
% bboxes = max(bboxes,1);
% 
% resizeScale = targetSize(maxDim)/imgSize(maxDim);
% 
% bboxes = bboxresize(round(bboxes),resizeScale);

if(isempty(bboxes))
    data = [];
    return;
end

bboxes = max(bboxes,1);

data{1} = im;
data{2} = bboxes;
data{3} = labels;
data{4} = masks;

end