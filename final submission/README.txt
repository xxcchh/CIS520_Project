The four we implemented were a generative model, discriminative model, semi-supervised dimensionality reduction model, and an instance based method with the text data. We used a 10-fold cross validation to calculate the training error. 

Generative Model

Our generative model was a Naive Bayes Model. We used the Matlab function, fitcnb() and set the distribution within the model as multinomial distribution. The model had a training error of 0.80. 

To run the model the Naive Bayes Model… 

Discriminative Model

The discriminative model we fit was a cross-validated SVM classifier using Bayesian Optimization. 


Instance Based Method

Semi-Supervised Dimensionality Reduction

The 