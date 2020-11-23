function  out = cocoAnnotationMATReader(filename, trainImgFolder)

% Copyright 2020 The MathWorks, Inc.

    data = load(filename);
    
    im = imread([trainImgFolder '/' data.imageName]);
    
    % For grayscale images repeat the image across the RGB channel
    if(size(im,3)==1)
        im = repmat(im, [1 1 3]);
    end
    out{1} = im;
    out{2} = data.bbox;
    out{3} = data.label;
    out{4} = data.masks;


end