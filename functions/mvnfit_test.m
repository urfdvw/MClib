% test mvnfit() function
clear
close all
clc
%% data
mu = [2 3];
sigma = [1 1.5; 1.5 3];
N = 10000;
R = mvnrnd(mu,sigma,N);
%% test
[mu,sigma] = mvnfit(R,rand(1,N))
[mu,sigma] = mvnfit(R,ones(1,N))
[mu,sigma] = mvnfit(R)
%% corner case
N = 1;
R = mvnrnd(mu,sigma,N);
[mu,sigma]=mvnfit(R,ones(1,N))