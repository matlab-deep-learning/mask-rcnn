function [AG] = groupProposalsByImageIndex(A,N)
% groupProposalsByImageIndex - helper function to group numProposalsx5 proposals 
% by the batch id specified at the 5th index. The output is a cell array of
% numBatchIndices x 1. Each cell holding Mix4 proposals for the ith image
% in the batch.

% Copyright 2020 The MathWorks, Inc.

    % Group by indices.
    ia = gather(A(:,5)); % gather: gpuArray does not support accumarray w/ @(x){x}.
    iag = accumarray(ia, 1:numel(ia), [N 1], @(x){x});

    AG = cell(N,1);
    for i = 1:numel(iag)
        if isempty(iag{i})
            AG{i} = zeros(0,5,'like',A);
        else        
            AG{i} = A(iag{i},:);
        end
    end
end