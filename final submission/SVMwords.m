clear;
load ../train_set/words_train
%crossvalidation
%{
trainsetpart=make_xval_partition(size(X,1),9);
trainsetpart=trainsetpart';

for j=1:1
    trainset=X(trainsetpart~=j,:);
    trainLabel=Y(trainsetpart~=j);
    testset=X(trainsetpart==j,:);
    testLabel=Y(trainsetpart==j);
    svm=fitcsvm(full(trainset),full(trainLabel),'KernelFunction','linear','Prior','uniform');
    c=predict(svm,full(testset));
    accuracy=mean(c==testLabel);
end
%}
svm=fitcsvm(full(X),full(Y),'KernelFunction','linear','Prior','uniform');
c=predict(svm,full(X));
accuracy=mean(c==Y);