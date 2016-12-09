function [CV_error, SVMmodel] = SVMwords(train_color,words_train)
%crossvalidation
k=10; %k fold Cross-Validation
trainsetpart=make_xval_partition(size(train_color,1),k);
trainsetpart=trainsetpart';

for j=1:k
    y=0;
    trainset=train_color(trainsetpart~=j,:);
    trainLabel=Y(trainsetpart~=j);
    testset=train_color(trainsetpart==j,:);
    testLabel=Y(trainsetpart==j);
    svm=fitcsvm(trainset,trainLabel,'KernelFunction','gaussian','Prior','uniform');
    y=predict(svm,testset);
    accuracy(j,1)=mean(y==testLabel);
end
CV_error=1-mean(accuracy);
SVMmodel=fitcsvm(train_color,full(Y),'KernelFunction','gaussian','Prior','uniform');
end