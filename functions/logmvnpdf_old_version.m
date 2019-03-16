function p=logmvnpdf_old_version(x,mu,C)
% Log Multivariable Gaussian PDF
% 
% inputs:
%     samples: x: N*D or D*N, if N==D then D*N
%     mean: mu: 1*D or D*1
%     covariance matrix: C: D*D
% 
% outputs:
%     probability: p: N*1 if x N*D, 1*N if x D*N

%% input variable shapes
% convert mu into D*1
if length(mu) ~= numel(mu)
    error('mu should be a vector')
end
mu = mu(:);
D = length(mu);
% convert x into D*N
if size(x,1) == D
    x_transposed = 0;
elseif size(x,2) == D
    x_transposed = 1;
    x = x';
else
    error('size of x does not match the distribution dimension')
end
N = size(x,2);

%% compute the log likelihood
p1 = -1/2*log(det(2*pi*C));
invC = inv(C);
p = zeros(1,N);
for n = 1:N
    p(n) = p1 - 1/2 * (x(:,n)-mu)' * invC * (x(:,n)-mu);
end

%% output shape
if x_transposed
    p = p';
end