clear;
load ../train_set/words_train
% K = 5;
K = 449;
% n = size(X, 1);
% partition = crossvalind('Kfold', n, 5);
% c = zeros(10, 1);
% for i = 1:10
%    testidx = (partition == i); 
%    trainidx = ~testidx;
%    Xtrain = X(trainidx, :);
%    Xtest = X(testidx, :);
%    Ytrain = Y(trainidx, :);
%    Ytest = Y(testidx, :);
%    [idx, ~] = knnsearch(Xtrain, Xtest, 'dist','jaccard', 'k', K);
%    Ypred = zeros(size(Ytest));
%    for j = 1:length(Ypred)
%         Ypred(j) = mean(Ytrain(idx(j, :))) > 0.5;
%    end
%    c(i) = mean(Ytest ~= Ypred);
% end
% err = mean(c);
Mdl = fitcknn(X,Y,'Distance','spearman',...
    'NumNeighbors', K, 'KFold', 10);
err = kfoldLoss(Mdl);


