classdef RegionProposal < nnet.layer.Layer

% Copyright 2020 The MathWorks, Inc.

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
        
    end
    
    properties        
        
        % MinSize The minimum proposal box size. Boxes smaller than MinSize
        % are discarded.
        MinSize
        
        % MaxSize The maximum proposal box size. Boxes larger than MaxSize
        % are discarded.
        MaxSize
        
        % ScaleFactor ([sx sy]) Used to scale boxes from feature map to
        %             image.
        ScaleFactor
        
        % ImageSize The size of the image for which boxes are proposed.
        ImageSize
        
        % NumStrongestRegions The number of strongest proposal boxes to
        %                     return.
        NumStrongestRegions
        
        NormalizedBoxes
    end
    
    properties(SetAccess = private)
        
        % OutputNames
        %OutputNames = {'out'}
        
        % HasSizeDetermined True for layers with size determined.
        HasSizeDetermined = true;
        
        % AnchorBoxes The anchor boxes used as reference for final
        %             detections.
        AnchorBoxes
        
        % ProposalsOutsideImage Whether proposals outside image should be
        %                       clipped or discarded. Set to 'clip' to
        %                       clip. Otherwise boxes are discarded.
        ProposalsOutsideImage
        
        % OverlapThreshold The overlap threshold used for NMS.
        OverlapThreshold                
        
        % NumStrongestRegionsBeforeProposalNMS
        % The number of strongest proposals to keep prior to running NMS.
        % This help reduce the number of boxes that need to be processed.
        NumStrongestRegionsBeforeProposalNMS
        
        % MinScore The minimum proposal score required for a box to be
        %          output by this layer.
        MinScore               
        
        % The standard dev and mean of the boxes. Used when applying the
        % box regression offsets.
        RPNBoxStd
        RPNBoxMean                
        
        % BoxFilterFcn A function handle. The specified function is used to
        % filter out boxes based on MinSize and MaxSize. A function handle
        % allows customization of box filtering. For example, a monoCamera
        % based filtering is used in ADST.
        BoxFilterFcn
        
    end
    
    methods
        function this = RegionProposal(name, anchorBoxes, params)
            
            this.AnchorBoxes = anchorBoxes;
            this.Name = name;
            this.InputNames = {'scores','boxDeltas'};
            this.ImageSize = params.ImageSize;
            this.RPNBoxStd = params.RPNBoxStd;
            this.RPNBoxMean = params.RPNBoxMean;
            this.MinSize = params.MinSize;
            this.MaxSize = params.MaxSize;
            this.NumStrongestRegions = params.NumStrongestRegions;
            this.NumStrongestRegionsBeforeProposalNMS = params.NumStrongestRegionsBeforeProposalNMS;
            this.ProposalsOutsideImage = params.ProposalsOutsideImage;
            this.MinScore = params.MinScore;
            this.OverlapThreshold = params.OverlapThreshold;
            this.ScaleFactor = params.ScaleFactor;
            this.BoxFilterFcn = params.BoxFilterFcn;
            this.NormalizedBoxes = true;
            
        end
        
        function allBoxes = predict(this, classificationBatch, regressionBatch)
            
            % Return bboxes in [x1 y1 x2 y2 idx] format. The boxes are in
            % the image space.
            % allBoxes will be returned as - 1 x 1 x C x B
            %  
            
            featureSize = size(classificationBatch);
            
            if isempty(this.ImageSize)
                imageSize = ceil(featureSize(1:2) .* fliplr(this.ScaleFactor));
            else
                imageSize = this.ImageSize;
            end
            
            % Get number of observations;
            N = size(classificationBatch,4);
            allBoxes = cell(N,1);
            for i = 1:N
                cls = classificationBatch(:,:,:,i);
                reg = regressionBatch(:,:,:,i);
                
                [bboxes, scores] = this.proposeImpl(cls, reg);
                
                bboxes = this.postProcessing(bboxes, scores, imageSize);
                
                bboxes = this.xywhToX1Y1X2Y2(bboxes);
                
                % Associate batch indices with each set of boxes.
                numBoxes = size(bboxes,1);
                allBoxes{i} = [bboxes repelem(i,numBoxes,1)];
            end
            
            allBoxes = single(vertcat(allBoxes{:}));
            
            if(isempty(allBoxes))
                warning('Empty Proposals. Skipping training step');
            end
            % convert to 5XM format
            allBoxes = allBoxes';            
            
            % Reshape to [1 1 5 M] format
%             allBoxes = reshape(allBoxes, [1 1 size(allBoxes)]);
        end

        function [dCls, dReg] = ...
            backward(~, classification, regression, ~, ~, ~)
        
            dCls = zeros(size(classification),'like',classification);
            dReg = zeros(size(regression),'like',regression);
            
        end
        
    end
    
    methods(Hidden)
        %------------------------------------------------------------------
        function [bboxes, scores] = proposeImpl(this, cls, reg)
            % Input cls are scores from the rpn cls conv layer 
            %   size(cls): [H W 2*NumAnchors numObs]
            %
            % Input reg are scores from the rpn reg conv layer
            %   size(reg): [H W 4*NumAnchors numObs]
            %
            % Output bboxes in [x y w h] format. In image space.
            
            % anchor boxes in [width height]
            aboxes = fliplr(this.AnchorBoxes);
            
            [fmapSizeY,fmapSizeX,numAnchorsTimesTwo,~] = size(cls);
            [y,x,k] = ndgrid(1:single(fmapSizeY), 1:single(fmapSizeX),1:size(aboxes,1));
            x = x(:);
            y = y(:);
            k = k(:);
            widthHeight = aboxes(k,:);
            
            % Box centers in image space (spatial coordinates).
            sx = 1/this.ScaleFactor(1);
            sy = 1/this.ScaleFactor(2);
            xCenter = x * sx + (1-sx)/2;
            yCenter = y * sy + (1-sy)/2;
            
            halfWidthHeight = widthHeight / 2 ;
            
            % top left in spatial coordinates
            x1 = xCenter - halfWidthHeight(:,1);
            y1 = yCenter - halfWidthHeight(:,2);
            
            % Convert to pixel coordinates.
            x1 = floor(x1 + 0.5);
            y1 = floor(y1 + 0.5);
            
            % anchor boxes as [x y w h]
            bboxes = [x1 y1 widthHeight];
            
            % Regression input is [H W 4*NumAnchors], where the 4
            % box coordinates are consecutive elements. Reshape data to
            % [H W 4 NumAnchors] so that we can gather the regression
            % values based on the max score indices.
            reg = reshape(reg, size(reg,1), size(reg,2), 4, []);
            
            % permute reg so that it is [H W NumAnchors 4]
            reg = permute(reg, [1 2 4 3]);
            
            % reshape reg so that it is [H*W*NumAnchors 4]
            reg = reshape(reg,[],4);
            
            % Apply box target normalization
            reg = (reg .* this.RPNBoxStd) + this.RPNBoxMean;
            
            bboxes = fastRCNNObjectDetector.applyReg(bboxes, reg);
            
            numAnchors = numAnchorsTimesTwo / 2;
            scores = reshape(cls(:,:,1:numAnchors,:),[],1);
            
            [bboxes, scores] = fastRCNNObjectDetector.removeInvalidBoxesAndScores(bboxes, scores);
            
        end
    end
    
    methods(Access = private)
        %------------------------------------------------------------------
        function [bboxes, scores] = postProcessing(this,bboxes,scores, imageSize)
            % Post processing is only supported on the CPU.
            %bboxes = gather(bboxes);
            %scores = gather(scores);
            
            if strcmp(this.ProposalsOutsideImage,'clip')
                bboxes = this.clipBBox(bboxes,imageSize);
                
            else
                H = imageSize(1);
                W = imageSize(2);
                outside =  bboxes(:,1) < 1 | bboxes(:,2) < 1 | (bboxes(:,1)+bboxes(:,3)-1) > W | (bboxes(:,2)+bboxes(:,4)-1) > H;
                bboxes(outside,:) = [];
                scores(outside) = [];
            end
            
            [bboxes, scores] = this.BoxFilterFcn(bboxes, scores, this.MinSize, this.MaxSize);
            
            % remove low scoring proposals
            if this.MinScore > 0
                lowScores = scores < this.MinScore;
                bboxes(lowScores,:) = [];
                scores(lowScores,:) = [];
            end
            
            [bboxes, scores] = rcnnObjectDetector.selectStrongestRegions(bboxes, scores, this.NumStrongestRegionsBeforeProposalNMS);
            
            [bboxes, scores] = selectStrongestBbox(bboxes, scores, ...
                'RatioType', 'Union', ...
                'OverlapThreshold', this.OverlapThreshold,...
                'NumStrongest', this.NumStrongestRegions); 
        end
        
        function boxes = xywhToX1Y1X2Y2(this, boxes)
            % Convert [x y w h] box to [x1 y1 x2 y2]. Input and output
            % boxes are in pixel coordinates. boxes is an M-by-4
            % matrix.
            boxes(:,3) = boxes(:,1) + boxes(:,3) - 1;
            boxes(:,4) = boxes(:,2) + boxes(:,4) - 1;
        end
        
        function clippedBBox = clipBBox(this, bbox, imgSize)

            %#codegen

            % bounding boxes are returned as doubles
            clippedBBox  = double(bbox);

            % The original bounding boxes all overlap the image. Therefore, a check to
            % remove non-overlapping boxes is not required.

            % Get coordinates of upper-left (x1,y1) and bottom-right (x2,y2) corners. 
            x1 = clippedBBox(:,1);
            y1 = clippedBBox(:,2);

            x2 = clippedBBox(:,1) + clippedBBox(:,3) - 1;
            y2 = clippedBBox(:,2) + clippedBBox(:,4) - 1;

            x1(x1 < 1) = 1;
            y1(y1 < 1) = 1;

            x2(x2 > imgSize(2)) = imgSize(2);
            y2(y2 > imgSize(1)) = imgSize(1);

            clippedBBox = [x1 y1 x2-x1+1 y2-y1+1];
        end
    end
end