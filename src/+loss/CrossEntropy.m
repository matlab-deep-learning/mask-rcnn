function loss = CrossEntropy( Y, T )
    % CrossEntropy   Return the cross entropy loss between estimate
    % and true responses averaged by the number of observations
    %
    % Syntax:
    %   loss = layer.forwardLoss( Y, T );
    %
    % Inputs:
    %   Y   Predictions made by network, M-by-N-by-numClasses-by-numAnchors
    %   T   Targets (actual values), M-by-N-by-numClasses-by-numAnchors  
    
    % Copyright 2020 The MathWorks, Inc.

    % Observations are encoded in T as non-zero values. T may
    % contain all zeros. Prevent divsion by zero preventing numObs
    % to be zero.
    numObservations = max(nnz(T),1); 

    % sum along numClasses
    loss = sum( T .* log(nnet.internal.cnn.util.boundAwayFromZero(Y)), 3);

    % sum all observations and average. Here all the
    % non-observations are also summed, but the loss for those is
    % zero, so it does not contribute.
    loss = -1/numObservations * sum(loss(:));                        
end