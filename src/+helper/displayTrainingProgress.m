function displayTrainingProgress (data, lossPlotter)

% Copyright 2020 The MathWorks, Inc.

disp("Epoch: " + data(1) + ", Iteration: " + data(2) + ", Loss: " + data(3) +  ", Learnrate: " + data(4));

addpoints(lossPlotter,double(data(2)),double(data(3)));

end