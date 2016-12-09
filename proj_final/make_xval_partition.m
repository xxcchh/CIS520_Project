function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE
k=0;
l=1;
part=zeros(1,n);
r=randperm(n);

x=floor(n/n_folds);
m=mod(n,n_folds);
r1=randperm(m);
save=zeros(n_folds,x);
for i=1:n_folds
    save(i,:)=i;
end
for i=1:n_folds
    for j=1:x
        k=k+1;
        part(r(k))=save(i,j);
    end
end
if m~=0
    for i=1:n
        if part(i)==0
            part(i)=r1(l);
            l=l+1;
        end
    end
end
    
