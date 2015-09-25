clear;
clc;
addpath('/home/hanjun/Software/liblinear/2.01/matlab');
addpath('/home/hanjun/Software/libsvm/3.20/matlab');
load bow-train_dev.mat;
trainy = double(trainy');
devy = double(devy');
methods = {'AdaBoostM2', 'LPBoost', 'RUSBoost', 'TotalBoost', 'Bag', 'Subspace'};
learners = {'Discriminant', 'Tree'};

model = fitensemble(trainx, trainy, methods{1}, 300, learners{1});

pred_y = predict(model, devx);

fprintf('acc=%.5f\n', mean(pred_y == devy));