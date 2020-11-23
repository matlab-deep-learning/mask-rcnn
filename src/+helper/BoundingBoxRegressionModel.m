classdef BoundingBoxRegressionModel < handle
% This class defines the bounding box regression model defined here:
%
% Girshick, Ross, et al. "Rich feature hierarchies for accurate object
% detection and semantic segmentation." Proceedings of the IEEE conference
% on computer vision and pattern recognition. 2014.

% Copyright 2020 The MathWorks, Inc.

    properties
        ModelX
        ModelY
        ModelW
        ModelH          
        
        Lambda
        Params
              
        IsTrained
    end
    
    %======================================================================
    methods
        function this = BoundingBoxRegressionModel(params)
            this.Lambda    = params.Lambda;
            this.Params    = {'Solver', 'dual', 'Regularization','ridge', 'ObservationsIn','columns'};                        
            this.IsTrained = false;
        end
        
        %------------------------------------------------------------------
        function [beta, bias] = getInitialWeights(~, featureLength)
            initValue = zeros(featureLength, 1);
            beta.X = initValue;
            beta.Y = initValue;
            beta.W = initValue;
            beta.H = initValue;
            
            bias.X = 0;
            bias.Y = 0;
            bias.W = 0;
            bias.H = 0;
        end
        
        %------------------------------------------------------------------ 
        function update(this, X, Y)            
            
            if this.IsTrained                
                [beta, bias] = getWeights(this);
            else
                [beta, bias] = getInitialWeights(this, size(X,1));
            end
            
            this.ModelX = fitrlinear(X, Y(:,1), 'Beta', beta.X, 'Bias', bias.X, 'Lambda', this.Lambda, this.Params{:});
            this.ModelY = fitrlinear(X, Y(:,2), 'Beta', beta.Y, 'Bias', bias.Y, 'Lambda', this.Lambda, this.Params{:});
            this.ModelW = fitrlinear(X, Y(:,3), 'Beta', beta.W, 'Bias', bias.W, 'Lambda', this.Lambda, this.Params{:});
            this.ModelH = fitrlinear(X, Y(:,4), 'Beta', beta.H, 'Bias', bias.H, 'Lambda', this.Lambda, this.Params{:});
                                       
            % After update is called once, the model is considered trained.
            this.IsTrained = true;
        end        
        
        %------------------------------------------------------------------
        function [beta, bias] = getWeights(this)
            beta.X = this.ModelX.Beta;
            beta.Y = this.ModelY.Beta;
            beta.W = this.ModelW.Beta;
            beta.H = this.ModelH.Beta;
            
            bias.X = this.ModelX.Bias;
            bias.Y = this.ModelY.Bias;
            bias.W = this.ModelW.Bias;
            bias.H = this.ModelH.Bias;
        end
        
        %------------------------------------------------------------------ 
        function fit(this, X, Y)              
            this.ModelX = fitrlinear(X, Y(:,1), 'Lambda', this.Lambda, this.Params{:});
            this.ModelY = fitrlinear(X, Y(:,2), 'Lambda', this.Lambda, this.Params{:});
            this.ModelW = fitrlinear(X, Y(:,3), 'Lambda', this.Lambda, this.Params{:});
            this.ModelH = fitrlinear(X, Y(:,4), 'Lambda', this.Lambda, this.Params{:});
        
            this.IsTrained = true;
        end               
        
        function g = apply(this, features, P)
            if ~isempty(features)
                [x, y, w, h] = predict(this, features);
                
                % center of proposals
                px = P(:,1) + floor(P(:,3)/2);
                py = P(:,2) + floor(P(:,4)/2);
                
                % compute regression value of ground truth box
                gx = P(:,3).*x + px; % center position
                gy = P(:,4).*y + py;
                
                gw = P(:,3) .* exp(w);
                gh = P(:,4) .* exp(h);
                
                % convert to [x y w h] format
                g = [ gx - floor(gw/2) gy - floor(gh/2) gw gh];
            else
                g = zeros(0,4);
            end
            
            g = round(g);
            
        end
        
        %------------------------------------------------------------------
        function [tx, ty, tw, th] = predict(this, features)
            tx = predict(this.ModelX, features, 'ObservationsIn', 'columns');
            ty = predict(this.ModelY, features, 'ObservationsIn', 'columns');
            tw = predict(this.ModelW, features, 'ObservationsIn', 'columns');
            th = predict(this.ModelH, features, 'ObservationsIn', 'columns');                        
        end
        
        %------------------------------------------------------------------    
        function [x, y] = getTrainingSamples(~, data, th)
            P = getProposals(data);
            G = getGroundTruth(data);
            F = getRegionFeatures(data);
                      
            [G, P, selected] = helper.BoundingBoxRegressionModel.selectBBoxesForTraining(G, P, th);
            
            x = F(:, selected);
            
            y = helper.BoundingBoxRegressionModel.generateRegressionTargets(G, P);
        
        end
    end
    
    %======================================================================
    methods(Static)        
        function y = generateRegressionTargets(G, P)
            % Create regression targets.
            % center of proposal
            px = P(:,1) + floor(P(:,3)/2);
            py = P(:,2) + floor(P(:,4)/2);
            
            % center of gt
            gx = G(:,1) + floor(G(:,3)/2);
            gy = G(:,2) + floor(G(:,4)/2);
            
            tx = (gx - px)./P(:,3);
            ty = (gy - py)./P(:,4);
            tw = log(G(:,3)./P(:,3));
            th = log(G(:,4)./P(:,4));
            
            y = [tx ty tw th]; % observations in columns
        end

        %------------------------------------------------------------------
        function [targets, positiveIndex, negativeIndex, assignments] = generateRegressionTargetsFromProposals(...
                regionProposals, groundTruth, posOverlap, negOverlap, matchAllGroundTruth)

            [assignments, positiveIndex, negativeIndex] = ...
                helper.boxAssignmentUtils.assignBoxesToGroundTruthBoxes(...
                regionProposals, groundTruth, ...
                posOverlap, ...
                negOverlap, matchAllGroundTruth);

            % Create an array that maps ground truth box to positive
            % proposal box. i.e. params is the closest grouth truth box to
            % each positive region proposal.
            if isempty(groundTruth)
                targets = {[]};
            else
                G = groundTruth(assignments(positiveIndex), :);
                P = regionProposals(positiveIndex,:);

                % positive sample regression targets
                targets = helper.BoundingBoxRegressionModel.generateRegressionTargets(G, P);
                targets = {targets'}; % arrange as 4 by num_pos_samples
            end
        end

        %------------------------------------------------------------------ 
        function [G, P, L, selected] = selectBBoxesForTraining(G, P, L, th)
            % find proposals that overlap with gt by some threshold. these
            % are use for training.
            
            % Input L contains label of each ground truth.
            
            if isempty(G)
                iou = zeros(0,size(P,1));
            elseif isempty(P)
                iou = zeros(size(G,1),0);
            else
                iou = bboxOverlapRatio(G, P, 'union');
            end
            
            lower = th(1);
            upper = th(2);
             
            % find ground truth that overlaps the most with each
            % proposal. index of ground truth 
            [v,i] = max(iou,[],1);
                        
            selected = v >= lower & v <= upper;
            
            L = L(i,:); % assign ground truth label to each proposal
            L = L(selected,:); % select proposals within overlap range
            P = P(selected,:);            
            G = G(i(selected), :); % create an array that maps groundTruth to proposal, i.e. P(i,:) is assigned to G(i,:).
            
        end
    end
end
