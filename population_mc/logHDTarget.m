function logw = logHDTarget(x,mu,sigma)

logw1 = logmvnpdf(x,-mu,sigma);
logw2 = logmvnpdf(x,+mu,sigma);
logw = zeros(size(logw1));
for i = 1:size(logw1,1)
    logw(i) = logmean([logw1(i),logw2(i)]);
end