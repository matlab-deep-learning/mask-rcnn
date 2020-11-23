classdef rpnSoftmax < nnet.layer.Layer

% Copyright 2020 The MathWorks, Inc.
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
    end
    
    methods
        function layer = rpnSoftmax(name)
            layer.Name = name;
        end
        
        function [Z] = predict(~, X)
            
            [H, W, numAnchorsTimesTwo, numObservations] = size(X);            
            X = reshape(X, H*W, numAnchorsTimesTwo/2, 2, numObservations );
            
            X = X - max(X,[],3);
            X = exp(X);
            Z = X./sum(X,3);
        end

        function [dX] = backward(~, X, Z, dZ, ~)
            
            dotProduct = sum(Z.*dZ, 3);
            dX = dZ - dotProduct;
            dX = dX.*Z;
            [H, W, numAnchorsTimesTwo, numObservations] = size(X);
            dX = reshape(dX, H, W, numAnchorsTimesTwo, numObservations);
        
        end
    end
end