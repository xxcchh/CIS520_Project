%% Boosting
%{
X1 = full(X);
Y1 = full(Y); 

indices = crossvalind('Kfold',Y1,10);
total_error = 0;
for i = 1:10
    test = (indices == i); train = ~test;
    ens = fitensemble(X1(train,:),Y1(train,:),'Subspace',50,'KNN');
    predicted_labels = predict(ens,X1(test,:));
    
    fold_error = (sum(Y1(test,:) ~= predicted_labels))/length(Y1(test,:))

    total_error = total_error + fold_error;
end


T = 50;
indices = crossvalind('Kfold',Y1,5);
total_error = 0;
for i = 1:10
    test = (indices == i); train = ~test;
    ens = fitensemble(X1(train,:),Y1(train,:),'AdaBoostM1',T,'Tree');
    predicted_labels = predict(ens,X1(test,:));
    
    fold_error = (sum(Y1(test,:) ~= predicted_labels))/length(Y1(test,:))

    total_error = total_error + fold_error;
end

total_error = total_error + fold_error;

T = 100;
indices = crossvalind('Kfold',Y1,5);
total_error = 0;
for i = 1:10
    test = (indices == i); train = ~test;
    ens = fitensemble(X1(train,:),Y1(train,:),'TotalBoost',T,'Tree');
    predicted_labels = predict(ens,X1(test,:));
    
    fold_error = (sum(Y1(test,:) ~= predicted_labels))/length(Y1(test,:))

    total_error = total_error + fold_error;
end


%%'LogitBoost'
indices = crossvalind('Kfold',Y1,5);
total_error = 0;
for i = 1:5
    test = (indices == i); train = ~test;
    ens = fitensemble(X1(train,:),Y1(train,:), 'LogitBoost', T, 'Tree');
    predicted_labels = predict(ens,X1(test,:));
    
    fold_error = (sum(Y1(test,:) ~= predicted_labels))/length(Y1(test,:))

    total_error = total_error + fold_error;
end
%}
%GentleBoost
clear;
T = 300;
load ../train_set/words_train
X=full(X);
Y=full(Y);
ens = fitensemble(X,Y, 'GentleBoost', T, 'Tree');
%{
indices = crossvalind('Kfold',Y,10);
total_error = 0;
for i = 1:10
    test = (indices == i); train = ~test;
    ens = fitensemble(X(train,:),Y(train,:), 'GentleBoost', T, 'Tree');
    predicted_labels = predict(ens,X(test,:));
    
    fold_error = (sum(Y(test,:) ~= predicted_labels))/length(Y(test,:))

    total_error = total_error + fold_error;
end
total_error = total_error/10;
%}