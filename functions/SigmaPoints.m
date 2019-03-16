function x = SigmaPoints(mu, sigma)
% generate sigma points given mean and covariance

D = length(mu);
u = [eye(D); -eye(D)];
A = chol(sigma + 1e-10*eye(D), 'lower');
x = A*u + mu;