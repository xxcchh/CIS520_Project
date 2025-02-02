function [CV_error, GMModel] = PCA_GMM(train_color, words_train)
load train_color.mat;
load words_train.mat; 

%% PCA and GMM
X1 = train_color;
Y1 = full(Y);
indices = crossvalind('Kfold',Y1,10);
total_error = 0;

for i = 1:10
    test = (indices == i); train = ~test;
    [W, Z] = pca(X1(train,:), 'NumComponents', 30);
    
    %Project test data into new PC dimension
    meantrain = mean(X1(train,:), 1);
    testset_s = bsxfun(@minus,X1(test,:),meantrain);
    test_v =testset_s * W;

    % Run GMM
    options = statset('MaxIter',1000);
    GMModel = fitgmdist(Z,2, 'CovarianceType','diagonal', 'Options', options, 'RegularizationValue',0.0001);
    idx = cluster(GMModel,Z); 

    %Match cluster index to letter using majority vote
    val = zeros(2,1);
    for i = 1:2 %go through each component
    c_val = idx == i;  
    y=Y1(train);
    val(i)=mode(y(c_val)); %take mode of letter that comes up the most
    end
    
    % 1 = 0
    % 2 = 1

    %Run cluster on test dat
    test_val = cluster(GMModel,test_v); 
    test_val( test_val==1 )=0; 
    test_val( test_val==2 )=1;

    fold_error = (sum(Y1(test) ~= test_val))/length(Y1(test,:));
    total_error = total_error + fold_error;
end
CV_error = total_error/10; 

[W, Z] = pca(X1, 'NumComponents', 30);
GMM_mod = fitgmdist(Z,2, 'CovarianceType','diagonal', 'Options', options, 'RegularizationValue',0.0001);

save GMM_Model GMM_mod;
end 



    