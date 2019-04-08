function [ind] = indexSample(sample_n, W)
% Sample the index by the 
W(isnan(W)) = 0;
M = length(W);
try
    ind = randsample(M,sample_n,true,W);
catch
    max(W)
    W
    error('randsample error')
end
end
