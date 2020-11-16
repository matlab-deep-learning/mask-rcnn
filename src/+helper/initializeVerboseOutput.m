function initializeVerboseOutput(~)
% if options.Verbose
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
    disp("|=====================================================================================================================================================|")
    disp("|  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Validation  |  Mini-batch  |  Validation  |  Base Learning  | Train Time | Validation Time |")
    disp("|         |             |   (hh:mm:ss)   |   Accuracy   |   Accuracy   |     Loss     |     Loss     |      Rate       | (hh:mm:ss) |   (hh:mm:ss)    |")
    disp("|=====================================================================================================================================================|")
% end
end