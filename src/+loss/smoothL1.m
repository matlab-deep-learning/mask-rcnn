function loss = smoothL1( Y, T, W)
% Return the loss between the output obtained from
% the network(Y) and the expected output(T). The loss is weighted based on
% the weights stored in W.
%
% Inputs
%   Y - The output from the network
%   T - The target responses. 
%   W - The intance weights.
%
%   Y and T are the same size.
%
% Outputs
%   loss - the loss between Y and T. Set instance weights correctly
%   to achieve average loss. For example, W = 1/N, from most cases
%   where N is the number of observations.

% Copyright 2020 The MathWorks, Inc.


    X = Y - T;

    loss = zeros(size(X), 'like', X);

    one     = ones(1.0,'like',X);
    onehalf = cast(0.5,'like',X);

    % abs(x) < 1
    idx = (X > -one) & (X < one);
    loss(idx) = 0.5 * X(idx).^2;

    % x >= 1 || x <= 1
    idx = ~idx;
    loss(idx) = abs(X(idx)) - onehalf;
    
    if(~isempty(W))
        loss = W .* loss;
        loss = sum(loss(:));
    else
        loss = sum(loss(:))/numel(Y);
    end
            
    
end