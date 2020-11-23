function loss = SpatialCrossEntropy(Y, T, positiveIds)

% Copyright 2020 The MathWorks, Inc.

    if(any(positiveIds))
        
        positiveIds = find(positiveIds);
        numpixelsPerObs = size(Y,1)*size(Y,2)*size(Y,3);
        loss = crossentropy(Y(:,:,:,positiveIds), T(:,:,:, positiveIds),"TargetCategories", "independent");
        loss = loss/numpixelsPerObs;
    else
        loss = dlarray(0);
    end
end