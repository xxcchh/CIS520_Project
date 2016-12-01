%% load data 
load data.mat
%% test
Ypred = predict_labels(X);
%%
%feature use
n = size(X, 1);
[trainIdx, testIdx] = crossvalind('HoldOut', n, 0.3); 
Xtrain = X(trainIdx, :);
Xtest = X(testIdx, :);
Y = full(Y);
Ytrain = Y(trainIdx, :);
Ytest = Y(testIdx, :);
%% words
% svm
% ridge/lars
%% svm
addpath('./liblinear');
%%
best = train(Y, X, '-C -s 2');
svmModel = train(Y, X, sprintf('-c %f -s 0', best(1)));
save('svmModel.mat', 'svmModel');
% [Ypred, acc] = predict(Ytest, Xtest, svmModel);
% opts = struct('Optimizer','bayesopt', 'CVPartition',c,...
%     'AcquisitionFunctionName','expected-improvement-plus');
% svmModel = fitcsvm(X, Y, 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
%%


