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
addpath('./liblinear');
load svm
load nb
load ens
c(:,1)=predict(nb,full(word_counts));
c(:,2)=predict(svm,full(word_counts));
c(:,3)=predict(ens,full(word_counts));
C=sum(c,2);
C(C==1)=0;
C(C==2)=1;
C(C==3)=1;
Y_hat = full(C);
end
