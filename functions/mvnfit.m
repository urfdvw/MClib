function [mu,C]=mvnfit(x,w)
% fit weighted data into a multivariable normal distribution
% by mean and covariance
% 
% Inputs:
%     x, N*D matrix, N number of samples, D dimension of data
%     w, N*1 or 1*N vector, converted to N*1 in operation
% Outputs:
%     mu: 1*D: estimated mean
%     C: D*D: estimated covariance

[N,D] = size(x);
if nargin == 1
    w = ones(N,1);
end
w = w(:);
w = w / sum(w);
mu = w' * x;
C = (x-mu)' * (repmat(w,[1,size(x,2)]).*(x-mu));
if N == 0
  mu = mu * nan;
  C = C * nan;
end
