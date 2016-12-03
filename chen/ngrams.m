function [result] = ngrams(corpus, topWords, topNgrams)
% corpus is a text 
tokens = strsplit(corpus, ' ');
wordIndex = cellfun(@(x) find(strcmp(topWords, x) == 1), tokens, 'UniformOutput', false);
wordIndex = wordIndex(~cellfun('isempty', wordIndex));
l = numel(wordIndex);
result = [];
for i = 1 : (l-1)
    rowIndex = cell2mat(wordIndex(i));
    colIndex = cell2mat(wordIndex(i+1));
    idx = find((topNgrams(:,2) == rowIndex).*(topNgrams(:, 3) == colIndex)); 
    if ~isempty(idx)
        result = [result, topNgrams(idx, 1)];
    end
end
end

