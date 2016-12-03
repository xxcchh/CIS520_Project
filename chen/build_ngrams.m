function [ result ] = build_ngrams(raw_tweets)
load topWordsNgrams
n = size(raw_tweets, 1);
l = size(topNgrams, 1);
rowNoneZero = [];
colNoneZero = [];
valNoneZeros = [];
% cellfun(@(x) ngrams(char(x), topWords, l), raw_tweets)
for i = 1:n
    res = char(raw_tweets(i));
    idx = ngrams(res, topWords, topNgrams);     
    if ~isempty(idx)
        uniqueNumber = unique(idx);
        len = length(uniqueNumber);
        val = arrayfun(@(x) histc(idx, x), uniqueNumber);
        rowNoneZero = [rowNoneZero, i * ones(1, len)];
        colNoneZero = [colNoneZero, uniqueNumber];
        valNoneZeros = [valNoneZeros, val];
    end
end
result = sparse(rowNoneZero,colNoneZero,valNoneZeros, n, l);
end

