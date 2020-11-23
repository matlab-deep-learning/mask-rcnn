function W = regressionResponseInstanceWeights(assignedLabels, bgClass)

% Copyright 2020 The MathWorks, Inc.

for i = 1:length(assignedLabels)
    
bgIndex = strcmp(bgClass,categories(assignedLabels{i}));

W{i} = onehotencode(assignedLabels{i}, 2); %-> [N numclasses]

W{i}(isnan(W{i})) = 0;

W{i}(:, bgIndex) = []; 

posIdx = any(W{i},2);

W{i} = repelem(W{i}, 1, 4);

% Incorporate normalization term into instance weights. 
% Treat each positive as an observations. Times 4 because there are 4
% independent targets per positive observation.
numObservations = max(1,4*nnz(posIdx)); % prevent division by zero.
W{i} = (1./numObservations) * W{i};


end

% function W = regressionResponseInstanceWeights(assignedLabels, bgClass)
% 
% % labels - numProposalx1 categorical
% 
% for i = 1:length(assignedLabels)
%     
% bgIndex = strcmp(bgClass,categories(assignedLabels{i}));
% 
% W{i} = onehotencode(assignedLabels{i}', 1); %-> [numclasses+1 x N ]
% 
% W{i}(bgIndex,:) = []; 
% 
% posIdx = any(W{i},1);
% 
% W{i} = repmat(W{i}, 4, 1);
% 
% % Incorporate normalization term into instance weights. 
% % Treat each positive as an observations. Times 4 because there are 4
% % independent targets per positive observation.
% numObservations = max(1,4*nnz(posIdx)); % prevent division by zero.
% W{i} = (1./numObservations) * W{i};
% 
% 
% end