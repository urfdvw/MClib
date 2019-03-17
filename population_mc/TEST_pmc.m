% test PMC algorithms
clear
close all
clc
addpath('..\functions\')

%% target distribution
logTarget = @(x)logmvnpdf(x,[0,0],[2,0.6;0.6,1]);

%% define PMC sampler
N = 1000;
x0 = mvnrnd([0,0],10*[1,0;0,1],N);
plot2Dsample(x0)
pmc = PMCPlain(x0,logTarget);
pmc.setSigma(2);

%% sampling cycles
for i = 1:10
    pmc.sample(10)
    plot2Dsample(pmc.mu)
    [x_p, w_p] = pmc.posterior();
    [mu_c,C_c] = mvnfit(x_p,w_p)
end

%% visulization function
function plot2Dsample(x)
pause(0.1)
plot(x(:,1),x(:,2),'.')
axis([-10,10,-10,10])
end