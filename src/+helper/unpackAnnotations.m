function unpackAnnotations(trainCats, annotationfile, trainImgFolder, unpackLocation )
% Unpack COCO annotations to MAT files

% Copyright 2020 The MathWorks, Inc.

% Initialize the CocoApi object
coco = CocoApi(annotationfile);
  
% Get all image ids filtered based on train categories
catIds = coco.getCatIds('catNms',trainCats);
imgIds = coco.getImgIds('catIds',catIds);

% In-Memory datastore to manage imageIDs
imgID_DS = arrayDatastore(imgIds);
% Get image and ground truth data from imageIds
ds = transform(imgID_DS, @(x)helper.cocoAnnotationsFromID_preprocess(x{1}, coco,trainImgFolder, catIds));

i = 1;
while (ds.hasdata)
    
    data = read(ds);
    
    imageName = data{1};
    bbox = data{2};
    label = data{3};
    masks = data{4};
    
    imageName_Number = imageName(16:end-4);
            
    labelFilename = [unpackLocation '/label_' imageName_Number '.mat'];
    save(labelFilename, 'imageName', 'bbox', 'label', 'masks')
    i=i+1;
    
    
end

disp('Done!');

end