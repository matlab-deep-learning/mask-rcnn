function [bboxes, scores, labels] = filterBoxesAfterRegression(bboxes, scores, labels, imageSize)
    % Remove boxes that don't make sense after regression:
    % * boxes with non-positive width and height.
    % * boxes that have zero overlap with image.

    % Copyright 2020 The MathWorks, Inc.

    [bboxes, scores, labels] = fastRCNNObjectDetector.removeInvalidBoxesAndScores(bboxes, scores, labels);

    % Remove boxes that are completely outside the image.
    x1 = bboxes(:,1);
    y1 = bboxes(:,2);
    x2 = bboxes(:,3) + x1 - 1;
    y2 = bboxes(:,4) + y1 - 1;

    boxOverlapsImage = ...
        (x1 < imageSize(2) & x2 > 1) & ...
        (y1 < imageSize(1) & y2 > 1);

    bboxes = bboxes(boxOverlapsImage,:);
    scores = scores(boxOverlapsImage,:);
    labels = labels(boxOverlapsImage,:);            
end