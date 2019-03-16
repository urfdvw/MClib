function logw = w2logw(w, log_scale)
% convert weights to log weights
%
% Input:
%     w: vector, same shape as logw: normalized weight
%     log_scale: scaler: scaling factor used to reconstruct log weights
% 
% Output:
%     logw: vector: log weights
if nargin == 1
    log_scale = 0;
end
logw = log(w) + log_scale;