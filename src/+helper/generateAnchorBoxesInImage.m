function [anchorBoxes, anchorID] = generateAnchorBoxesInImage(...
    imageSize, featureMapSize, anchorBoxSizes, scaleFactor)
% Returns anchorBoxes in image space as a cell array. Each cell element 
% contains a set o% boxes generated at 1 scale and aspect ratio. Each box
% is centered within the receptive field.
% anchorID is a cell array. it is the index into the feature map where each
% anchor box maps. Two anchor boxes can map to the same feature map
% location. The ID helps prevent using the same feature for both positive
% and negative samples.

% Copyright 2020 The MathWorks, Inc.

% Generate box centers in feature map (spatial coordinates).
[X, Y] = meshgrid(1:featureMapSize(2),1:featureMapSize(1));
xCenterInFeatureMap = X(:);
yCenterInFeatureMap = Y(:);

sx = scaleFactor(1);
sy = scaleFactor(2);

% scale box center from feature space to image space.
[xCenterInImage, yCenterInImage] = scaleBoxCenter(xCenterInFeatureMap, yCenterInFeatureMap, sx,sy);

numAnchors = size(anchorBoxSizes,1);
anchorBoxes = cell(numAnchors,1);
anchorID = cell(numAnchors,1);

for i = 1:numAnchors
    
    sz = anchorBoxSizes(i,:);
    
    halfWidth = sz ./ 2;
    
    xCenter = xCenterInImage;
    yCenter = yCenterInImage;
    
    dim = repelem(fliplr(sz), size(xCenter,1), 1);
    
    % top left in spatial coordinates
    x1 = xCenter - halfWidth(2);
    y1 = yCenter - halfWidth(1);
    
    % Convert to pixel coordinates.
    x1 = floor(x1 + 0.5);
    y1 = floor(y1 + 0.5);
    
    % anchor boxes as [x y w h]
    boxes = [x1 y1 dim];
    
    % anchor IDs are just x,y location in feature map.
    ids = [xCenterInFeatureMap yCenterInFeatureMap];
    
    % Remove boxes that are outside the image.
    outside = (x1 < 1) | (y1 < 1) | ((x1 + dim(1) - 1) > imageSize(2)) | ((y1 + dim(2) - 1) > imageSize(1));
    
    boxes(outside,:) = [];
    ids(outside,:) = [];
    
    anchorID{i} = ids;
    
    anchorBoxes{i} = boxes;
    
    assert( all(ids(:) >= 1) )
    
    assert( all(ids(:,1) <= featureMapSize(2)) )
    
    assert( all(ids(:,2) <= featureMapSize(1)) )
    
    assert( size(anchorBoxes{i},1) == size(anchorID{i},1) );
end

%--------------------------------------------------------------------------
function [x, y] = scaleBoxCenter(x,y,sx,sy)
% x,y are in spatial units so 1,1 is center of pixel of top-left pixel.
x = x * sx + (1-sx)/2;
y = y * sy + (1-sy)/2;