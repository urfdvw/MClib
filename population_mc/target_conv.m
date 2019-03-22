function logw = target_conv(logtarget,sig_noise,t)
M = 200;
[N,D] = size(t);
u = randn([M,D])* sig_noise;
logw = zeros([N,1]);
for i = 1:N
    logw(i) = logmean(logtarget(t(i,:) + u));
end