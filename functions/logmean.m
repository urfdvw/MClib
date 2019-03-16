function logm = logmean(logw)
% find the log(m) if m is the average of all w(s) when log(w)s are given.
% This function is used when w(s) are too small that log(w) are more stable than w.
% 
% Inputs:
%     logw: vector: log of w
%     
% outputs:
%     logm: scaler: log of mean of w
[w, log_scale] = logw2w(logw);
m = mean(w);
logm = w2logw(m, log_scale);