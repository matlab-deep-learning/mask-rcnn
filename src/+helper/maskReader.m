function mask = maskReader(filename)
% MASKREADER loads masks from MAT file

out = load(filename);
mask = out.mask;