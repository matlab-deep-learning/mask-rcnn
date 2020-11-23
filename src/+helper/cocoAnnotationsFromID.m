function out = cocoAnnotationsFromID (id, coco, trainImgFolder, trainCatIds)

% Copyright 2020 The MathWorks, Inc.

imStruct = coco.loadImgs(id);
im = imread(fullfile(trainImgFolder,imStruct.file_name));

% For grayscale images repeat the image across the RGB channel
if(size(im,3)==1)
    im = repmat(im, [1 1 3]);
end

height = size(im,1);
width = size(im,2);

annIds = coco.getAnnIds('imgIds', id);
anns = coco.loadAnns(annIds);

numAnns = numel(anns);

% load bounding boxes from struct array into a numAnnsx4 matrix
bbox = (reshape([anns.bbox],[4 numAnns])') ;
%  index to 1 index
bbox(:,1:2) = bbox(:,1:2) +1;

% load categories as a numAnnsx1 array
cats = reshape([anns.category_id],[numAnns 1]);
iscrowd = reshape([anns.iscrowd],[numAnns 1]);
validcats = ismember(cats, trainCatIds) & ~iscrowd;

% filter out irrelevant labels and bboxes
cats = cats(validcats);
bbox = bbox(validcats,:);

categoryNames = cat(1,{coco.data.categories.name})';

trainCatNames = categoryNames(trainCatIds);

cats = categorical(cats, trainCatIds, trainCatNames);

mask = false(height, width, sum(validcats));

maskIdx = 1;
% load masks
for j = 1:numAnns
   
   if(~validcats(j))
       continue;
   end
   
   P = anns(j).segmentation; 
   if(isstruct(P))
       % dummy bbox
       bbox(j,:) = [1 1 1 1];
       continue;
   end
   maskInstance = false(height, width);
   for pidx=1:length(P)
       X = P{pidx}(1:2:end);
       Y = P{pidx}(2:2:end);
       maskInstance = maskInstance | poly2mask(X,Y, height, width); 
   end

   mask(:,:,maskIdx) = maskInstance;
   maskIdx = maskIdx+1;
end

out{1} = im;

if isempty(bbox)
    out{2} = [1 1 1 1];
    out{3} = missing;
    out{4} = false([height width]);
    return;
end

if(size(bbox,1)~=size(mask,3))
    a=10;
end
out{2} = bbox;
out{3} = cats;
out{4} = mask;