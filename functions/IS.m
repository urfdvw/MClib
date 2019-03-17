function [ind] = IS(varargin)
q=inputParser;
addParameter(q,'weights',nan);
addParameter(q,'logweights',nan);
addParameter(q,'sample_n',nan);
parse(q,varargin{:});
if ~isnan(q.Results.weights)
    W = q.Results.weights;
    W = W / sum(W(:));
end
if ~isnan(q.Results.logweights)
    logW = q.Results.logweights;
    logW = logW - max(logW);
    W = exp(logW);
    W = W / sum(W(:));
end
M = length(W);
if isnan(q.Results.sample_n)
    sample_n = M;
else
    sample_n = q.Results.sample_n;
end
ind = randsample(M,sample_n,true,W);
end
