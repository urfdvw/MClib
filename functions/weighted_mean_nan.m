function average = weighted_mean_nan(x,w)
% Weighted mean of vectors that ignores Nan data
% input
    % x: N*D
    % w: N*1 or 1*N
% output
    % average: 1:D
w = w/sum(w(:));
[N,D] = size(x);
average = zeros([1,D]);
for n = 1:N
    if any(isnan(x(n,:)))
    else
        average = average + w(n) * x(n,:);
    end
end
end