function mask = maskReader(filename)
% MASKREADER loads masks from MAT file

% Copyright 2020 The MathWorks, Inc.

out = load(filename);
mask = out.mask;