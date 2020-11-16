function downloadTrainedMaskRCNN(url,destination)
% The downloadTrainedMaskRCNN function downloads a pretrained 
% OpenPose network.
%
% Copyright 2020 The MathWorks, Inc.

filename = 'maskrcnn_pretrained_person_car.mat';
netDirFullPath = destination;
netFileFullPath = fullfile(destination,filename);

if ~exist(netFileFullPath,'file')
    fprintf('Downloading pretrained MaskRCNN network.\n');
    fprintf('This can take several minutes to download...\n');
    if ~exist(netDirFullPath,'dir')
        mkdir(netDirFullPath);
    end
    websave(netFileFullPath,url);
    fprintf('Done.\n\n');
else
    fprintf('Pretrained MaskRCNN network already exists.\n\n');
end
end