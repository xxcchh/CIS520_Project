clear;
load ../train_set/words_train
%crossvalidation
%{
trainsetpart=make_xval_partition(size(X,1),9);
trainsetpart=trainsetpart';
for j=1:9
    trainset=X(trainsetpart~=j,:);
    trainLabel=Y(trainsetpart~=j);
    testset=X(trainsetpart==j,:);
    testLabel=Y(trainsetpart==j);
    nb=fitNaiveBayes(trainset,trainLabel,'Distribution','mn');
    c=predict(nb,testset);
end
Accuracy=mean(c==testLabel);
%}
nb=fitNaiveBayes(X,Y,'Distribution','mn');
c=predict(nb,X);
accuracy=mean(c==Y);