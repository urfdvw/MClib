% Test logw2w() w2logw() and logmean()
clear
close all
clc

%% data
N = 1000;
w0 = rand(1,N); % unnormalized weights
logw0 = log(w0); % log weights
w1 = w0/sum(w0(:)); % normalized weights
logm0 = log(mean(w0)); % log of mean of the weights
%% test
[w, log_scale] = logw2w(logw0);
% w should be the same as w1
logw = w2logw(w, log_scale);
% logw should be the same as logw0
logm = logmean(logw0);
% logm should be the same as logm0