function displayTrainingProgress (data, lossPlotter)
disp("Epoch: " + data(1) + ", Iteration: " + data(2) + ", Loss: " + data(3) +  ", Learnrate: " + data(4));

addpoints(lossPlotter,double(data(2)),double(data(3)));

end