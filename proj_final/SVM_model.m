function [svmModel] = SVM_model( words_train )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
load words_train.mat;
addpath('./liblinear')
best = train(full(Y), X, '-C -s 2');
svmModel = train(full(Y), X, sprintf('-c %f -s 0', best(1)));
save('svmModel.mat', 'svmModel');
end

