classdef boxUtils
% A collection of box conversion and scaling utilities.
    
% Copyright 2020 The MathWorks, Inc.
    methods(Static)
        %------------------------------------------------------------------
        function boxes = xywhToX1Y1X2Y2(boxes)
            % Convert [x y w h] box to [x1 y1 x2 y2]. Input and output
            % boxes are in pixel coordinates. boxes is an M-by-4
            % matrix.
            boxes(:,3) = boxes(:,1) + boxes(:,3) - 1;
            boxes(:,4) = boxes(:,2) + boxes(:,4) - 1;
        end
        
        %------------------------------------------------------------------
        function boxes = x1y1x2y2ToXYWH(boxes)
            % Convert [x1 y1 x2 y2] boxes into [x y w h] format. Input and
            % output boxes are in pixel coordinates. boxes is an M-by-4
            % matrix.
            boxes(:,3) = boxes(:,3) - boxes(:,1) + 1;
            boxes(:,4) = boxes(:,4) - boxes(:,2) + 1;
        end
    end
end