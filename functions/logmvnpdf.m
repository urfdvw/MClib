function p=logmvnpdf(x,mu,C)
% Log Multivariable Gaussian PDF
% 
% inputs:
%     samples: x: N*D or D*N, if N==D then D*N
%     mean: mu: 1*D or D*1
%     covariance matrix: C: D*D
% 
% outputs:
%     probability: p: N*1 if x N*D, 1*N if x D*N

%% input shapes
% convert mu into 1*D
if length(mu) ~= numel(mu)
    error('mu should be a vector')
end
mu = mu(:)';
D = length(mu);
% convert x into N*D
if size(x,2) == D
    x_transposed = 0;
elseif size(x,1) == D
    x_transposed = 1;
    x = x';
else
    error('size of x does not match the distribution dimension')
end
N = size(x,1);

%% compute the log likelihood
% https://www.mathworks.com/matlabcentral/fileexchange/34064-log-multivariate-normal-distribution-function
const = -0.5 * D * log(2*pi);
xc = bsxfun(@minus,x,mu);
term1 = -0.5 * sum((xc / C) .* xc, 2); % N x 1
term2 = const - 0.5 * logdet(C);    % scalar
p = term1 + term2;

%% output shape
if x_transposed
    p = p';
end

%% sub function
function y = logdet(A)
U = chol(A);
y = 2*sum(log(diag(U)));
end
end