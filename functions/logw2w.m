function [w, log_scale] = logw2w(logw)
% convert log weights to weights
% 
% Input:
%     logw: vector: log weights
% 
% Output:
%     w: vector, same shape as logw: normalized weight
%     log_scale: scaler: scaling factor used to reconstruct log weights
log_scale = max(logw);
logw = logw - log_scale;
w = exp(logw);
sumw = sum(w);
w = w / sumw;
log_scale = log_scale + log(sumw);