function displayVerboseOutputEveryEpoch(start,learnRate,epoch,iteration,lossTrain)

% Copyright 2020 The MathWorks, Inc.

    D = duration(0,0,toc(start),'Format','hh:mm:ss');
        
    lossTrain = gather(extractdata(lossTrain));
    lossTrain = compose('%.4f',lossTrain);
    
    learnRate = compose('%.4f',learnRate);
    
    disp("| " + ...
        pad(string(epoch),7,'both') + " | " + ...
        pad(string(iteration),11,'both') + " | " + ...
        pad(string(D),14,'both') + " | " + ...
        pad(string(lossTrain),12,'both') + " | " + ...
        pad(string(learnRate),15,'both') + " | " )
end