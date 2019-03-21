function [ind] = indexSample(sample_n, W)
% Sample the index by the 
M = length(W);
ind = randsample(M,sample_n,true,W);
end
