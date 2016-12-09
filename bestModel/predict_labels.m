function [Y_hat] = predict_labels(word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets)
% Inputs:   word_counts     nx10000 word counts features
%           cnn_feat        nx4096 Penultimate layer of Convolutional
%                               Neural Network features
%           prob_feat       nx1365 Probabilities on 1000 objects and 365
%                               scene categories
%           color_feat      nx33 Color spectra of the images (33 dim)
%           raw_imgs        nx30000 raw images pixels
%           raw_tweets      nx1 cells containing all the raw tweets in text
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 0 for sad)
load svm
load nb
load ens
load SentimentalResultsFinal
% for i=1:size(word_counts)
%     c(i,1)=sum(sentimentalResult(find(word_counts(i,:)==1),1));
%     c(i,2)=sum(sentimentalResult(find(word_counts(i,:)==1),2));
%     c(i,3)=sum(sentimentalResult(find(word_counts(i,:)==1),3));
% end
% for i=1:size(c)
%     if c(i,1)>0 && c(i,3)==0
%             p(i,1)=0;
%     elseif c(i,3)>0 && c(i,1)==0
%             p(i,1)=1;
%     else
%             p(i,1)=1;
%     end
% end
% word_counts(word_counts>0)=1;
% P(:,1)=predict(nb,full(word_counts(p==1,:)));
% P(:,2)=predict(svm,full(word_counts(p==1,:)));
% P(:,3)=predict(ens,full(word_counts(p==1,:)));
% MP=sum(P,2);
% MP(MP==1)=0;
% MP(MP==2)=P(MP==2,1);
% MP(MP==3)=1;
% k=1;
% for l=1:size(p)
%     if p(l,1)==1
%         p(l,1)=MP(k,1);
%         k=k+1;
%     end
% end
% Y_hat = full(p);
n = size(word_counts, 1);
selectidx = ones(n, 1);
Y_hat = zeros(n, 1);
for i = 1:n
    [~, tempidx] = find(word_counts(i, :));
    pos = sentimentalResultFinal(tempidx, 3);
    neg = sentimentalResultFinal(tempidx, 1);
    if ~isempty(find(neg == 1)) && isempty(find(pos == 1))
        Y_hat(i) = 0;
        selectidx(i) = 0;
    end
    if ~isempty(find(pos == 1)) && isempty(find(neg == 1))
        Y_hat(i) = 1;
        selectidx(i) = 0;
    end
end
X = word_counts(selectidx == 1, :);
Y_pred = Y_hat(selectidx == 1);
P(:,1) = predict(nb,full(X));
P(:,2) = predict(svm,full(X));
P(:,3) = predict(ens,full(X));
MP = sum(P,2);
Y_pred(MP == 1) = 0;
Y_pred(MP == 2) = P(MP == 2, 1);
Y_pred(MP == 3) = 1;
Y_hat(selectidx == 1) = Y_pred;
end
