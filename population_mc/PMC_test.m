% test PMC algorithms
clear
close all
clc
addpath('..\functions\')

%% target distribution
logTarget = @(x)logmvnpdf(x,[0,0],[2,0.6;0.6,1]);

%% define PMC sampler
test_case = 3;

if test_case == 0 % original class, each population just 1 sample
    N = 1000;
    mu0 = mvnrnd([0,0],10*[1,0;0,1],N);
    pmc = PMCPlain(mu0,logTarget);
end
if test_case == 1 % each population just 1 sample
    N = 1000;
    mu0 = mvnrnd([0,0],10*[1,0;0,1],N);
    pmc = PMC(logTarget,mu0,1);
end

if test_case == 2 % global resampling (default resdampling)
    N = 100;
    mu0 = mvnrnd([0,0],10*[1,0;0,1],N);
    pmc = PMC(logTarget,mu0,10);
end

if test_case == 3 % local resampling
    N = 100;
    mu0 = mvnrnd([0,0],10*[1,0;0,1],N);
    pmc = PMC(logTarget,mu0,10);
    pmc.resample_method = 'local';
end

pmc.setSigma(2);
%% sampling cycles
for i = 1:10
    plot2Dsample(pmc.mu)
    pmc.sample()
    [x_p, w_p] = pmc.posterior();
    [mu_c,C_c] = mvnfit(x_p,w_p)
end
plot2Dsample(pmc.mu)

%% visulization function
function plot2Dsample(x)
pause(0.1)
plot(x(:,1),x(:,2),'.')
axis([-10,10,-10,10])
end