%% load data 
load data.mat
%% test
Ypred = predict_labels(word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets);
%%
%feature use
n = size(X, 1);
[trainIdx, testIdx] = crossvalind('HoldOut', n, 0.3); 
Xtrain = X(trainIdx, :);
Xtest = X(testIdx, :);
Y = full(Y);
Ytrain = Y(trainIdx, :);
Ytest = Y(testIdx, :);
%% methods
% svm
% ridge/lars
% naive bayes
% boost 
%% add ngrams 
idxMat = raw_tweets_train{1,1};  
idxTweets = tweet_ids;
raw_tweets = raw_tweets_train{1, 2};
%% find raw text
tweetsTrain = cell(n, 1);
for i = 1:n
    idx = find(idxMat == idxTweets(i));%there is one duplicate on 7 and 
    idx = idx(1);
    tweetsTrain{i} = char(raw_tweets(idx)); 
end
%% ngrams
Xngrams = build_ngrams(tweetsTrain);
Xnew = [X'; Xngrams']'; 
%% svm
addpath('./liblinear')
best = train(full(Y), X, '-C -s 2');
svmModel = train(full(Y), X, sprintf('-c %f -s 0', best(1)));
save('svmModel.mat', 'svmModel');
% c = cvpartition(n, 'KFold', 10);
% opts = struct('Optimizer','bayesopt', ...
%     'AcquisitionFunctionName','expected-improvement-plus');
% svmModel = fitcsvm(full(X), full(Y), 'KernelFunction','rbf',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
%%
% ridge 
k = 0:1e-5:5e-3;
b = ridge(Y, Xnew, k);
%%

