function bboxes = applyRegression(boxIn,reg, minSize, maxSize)
% APPLYREGRESSION applies box regressions to the input boxes.
%
% Inputs:
% boxIn       - Input bounding boxes/proposals Nx4 (formated as [x y w h])
% reg         - regression deltas conputed by the RCNN family of networks [dx dy dw dh]
% min/maxSize - Minimum and maximum size of the boxes.
%
% Output:
% bboxes      - Boxes after regression - Nx4 (formated as [x y w h])

% Copyright 2020 The MathWorks, Inc.

    x = reg(:,1);
    y = reg(:,2);
    w = reg(:,3);
    h = reg(:,4);

    % center of proposals
    px = boxIn(:,1) + floor(boxIn(:,3)/2);
    py = boxIn(:,2) + floor(boxIn(:,4)/2);

    % compute regression value of ground truth box
    gx = boxIn(:,3).*x + px; % center position
    gy = boxIn(:,4).*y + py;

    gw = boxIn(:,3) .* exp(w);
    gh = boxIn(:,4) .* exp(h);

    if nargin > 2
        % regression can push boxes outside user defined range. clip the boxes
        % to the min/max range. This is only done after the initial min/max size
        % filtering.
        gw = min(gw, maxSize(2));
        gh = min(gh, maxSize(1));

        % expand to min size
        gw = max(gw, minSize(2));
        gh = max(gh, minSize(1));
    end

    % convert to [x y w h] format
    bboxes = [ gx - floor(gw/2) gy - floor(gh/2) gw gh];

    bboxes = double(round(bboxes));

end