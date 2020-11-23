function initializeVerboseOutput(~)

% Copyright 2020 The MathWorks, Inc.

    disp(" ")
    if canUseGPU
        disp("Training on GPU.")
    else
        disp("Training on CPU.")
    end
    p = gcp('nocreate');
    if ~isempty(p)
        disp("Training on parallel cluster '" + p.Cluster.Profile + "'. ")
    end
    %disp("MiniBatchSize:" + string(options.MiniBatchSize));
    %disp("Classes:" + join(string(options.Classes), ","));
    disp("|=========================================================================|")
    disp("|  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Base Learning  |")
    disp("|         |             |   (hh:mm:ss)   |     Loss     |      Rate       |")
    disp("|=========================================================================|")
end