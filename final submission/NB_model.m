function [CV_error, NBModel] = NB_model(train_color, words_train)
load train_color.mat;
load words_train.mat; 

%Cross validation 
X1 = full(X);
Y1 = full(Y);
indices = crossvalind('Kfold',Y1,10);
total_error = 0;

for i = 1:5
    test = (indices == i); train = ~test;
    mod = fitcnb(X1(train,:),Y1(train,:),'Distribution','mn');
    predicted_labels = predict(mod ,X1(test,:));
    
    fold_error = (sum(Y1(test,:) ~= predicted_labels))/length(Y1(test,:))

    total_error = total_error + fold_error;
end
CV_error = total_error/10; 

NB_Model_final = fitcnb(X1(train,:),Y1(train,:), 'Distribution','mn');

save NBModel NB_Model_final;
