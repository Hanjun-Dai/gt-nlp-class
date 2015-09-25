clear;
clc;
addpath('/home/hanjun/Software/liblinear/2.01/matlab');
addpath('/home/hanjun/Software/libsvm/3.20/matlab');
load bow-train_dev.mat;
trainy = double(trainy');
devy = double(devy');

% model = svmtrain(trainy, sparse(trainx), '-s 1 -t 2 -g 0.0002');
% [pred_label, acc, ~] = svmpredict(devy, sparse(devx), model);

model = train(trainy, sparse(trainx), '-s 2 -c 0.0174 -q');
[pred_dev, ~, ~] = predict(devy, sparse(devx), model);
length(find(pred_dev == 0)) / length(pred_dev)

testy = zeros(size(testx, 1), 1);
[pred_test, ~, ~] = predict(testy, sparse(testx), model);
length(find(pred_test == 0)) / length(pred_test)

fid = fopen('kaggle_submit', 'w');

fprintf(fid, 'Id,Prediction\n');

for i = 1 : length(pred_test)
    fprintf(fid, 'test-%d,%d\n', i - 1, pred_test(i));
end

for i = 1 : length(pred_dev)
    fprintf(fid, 'dev-%d,%d\n', i - 1, pred_dev(i));
end

fclose(fid);