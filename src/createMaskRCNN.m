function dlnet = createMaskRCNN(numClasses, params, featureExtractor) 
% Create Mask RCNN network


validatestring(featureExtractor, {'resnet101', 'resnet50'});
% Create FasterRCNN network and modify it to a MaskRCNN
lgraph = fasterRCNNLayers(params.ImageSize, numClasses, params.AnchorBoxes,  featureExtractor);

switch(featureExtractor)
    case 'resnet50'
        detectorFeatureLayer = 'activation_49_relu';
        inputLayerName = 'input_1';
    case 'resnet101'
        detectorFeatureLayer = 'res5c_relu';
        inputLayerName = 'data';
    otherwise
        error('Unsupported feature extraction network');
end

inputLayer = imageInputLayer(params.ImageSize,'Normalization', 'rescale-symmetric', 'Max', 255, 'Min', 0, 'name','input');

lgraph = lgraph.replaceLayer(inputLayerName, inputLayer);

% Drop loss layers
lgraph = lgraph.removeLayers(lgraph.OutputNames);

rpnSftmax = layer.RPNSoftmax('rpnSoftmax');
% Swap RPN softmax with the custom layer
lgraph = lgraph.replaceLayer('rpnSoftmax', rpnSftmax);

% Add Mask Head to Faster RCNN
maskHead = createMaskHead(numClasses, params);

lgraph = lgraph.addLayers(maskHead);

lgraph = lgraph.connectLayers(detectorFeatureLayer, 'mask_tConv1');

% Replace RegionProposalLayer with custom RPL
customRegionProposal = layer.RegionProposal('rpl', params.AnchorBoxes, params);
lgraph = lgraph.replaceLayer('regionProposal', customRegionProposal);

% Replace roiMaxpooling with roiAlign
roiAlign = roiAlignLayer([14 14], 'Name', 'roiAlign', 'ROIScale', params.ScaleFactor(1));
lgraph = lgraph.replaceLayer('roiPooling', roiAlign);

% convert to dlnet
dlnet = dlnetwork(lgraph);
    
    
end
    
    
function layers = createMaskHead(numClasses, params)

    if(params.ClassAgnosticMasks)
        numMaskClasses = 1;
    else
        numMaskClasses = numClasses;
    end

    tconv1 = transposedConv2dLayer(2, 256,'Stride',2, 'Name', 'mask_tConv1' );

    conv1 = convolution2dLayer(1, numMaskClasses, 'Name', 'mask_Conv1','Padding','same' );

    sig1 = sigmoidLayer('Name', 'mask_sigmoid1');

    layers = [tconv1 conv1 sig1];  
end

