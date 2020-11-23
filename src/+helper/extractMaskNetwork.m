function maskNet = extractMaskNetwork(net)
% Extract the Mask subnetwork from a pre-trained maskrcnn network

% Copyright 2020 The MathWorks, Inc.

    roiInput = imageInputLayer([5 1000 1], 'Name', 'roiInput', 'Normalization', 'none');
    
    featureInput = imageInputLayer([50 50 1024], 'Name', 'featureInput', 'Normalization', 'none');

    
    lgraph = layerGraph(net);
    
    % To extract mask subnetwork, remove all backbone layers, rpn layers
    % and bbox regression and classification heads
    
    for i = 1: numel(lgraph.Layers)
        backboneLayers{i} = lgraph.Layers(i).Name;
        if (strcmp(lgraph.Layers(i).Name, 'res4b22_relu'))
            break;
        end
    end
    
    rpnLayers = {'rpnConv3x3', 'rpnRelu', 'rpnConv1x1BoxDeltas', 'rpnConv1x1ClsScores', 'rpnSoftmax','rpl'};
    
    bboxHeads = {'pool5', 'rcnnFC', 'rcnnSoftmax', 'fcBoxDeltas' };
    
    layersToRemove = [backboneLayers, rpnLayers, bboxHeads];
    
    for idx = 1:numel(layersToRemove)
        lgraph = lgraph.removeLayers(layersToRemove{idx});
    end

    % Add input layers
    lgraph = lgraph.addLayers(roiInput);
    lgraph = lgraph.addLayers(featureInput);
    
    lgraph = lgraph.connectLayers('roiInput', 'roiAlign/roi');
    lgraph = lgraph.connectLayers('featureInput', 'roiAlign/in');
    
    maskNet = dlnetwork(lgraph);

end