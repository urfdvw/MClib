% test PMC algorithms
clear
close all
clc
addpath('..\functions\')

%% target distribution
pdf_case = 2;

if pdf_case == 1
    logTarget = @(x)logmvnpdf(x,[-10,-10],[2,0.6;0.6,1]);
elseif pdf_case == 2
    mu = [-10,-10;
        0,16;
        13,8;
        -9,7;
        14,-14];
    sigma(:,:,1) = [2,0.6;0.6,1];
    sigma(:,:,2) = [2,-0.4;-0.4,2];
    sigma(:,:,3) = [2,0.8;0.8,2];
    sigma(:,:,4) = [3,0;0,0.5];
    sigma(:,:,5) = [2,-0.1;-0.1,2];
    gm = gmdistribution(mu,sigma);
    logTarget = @(x) log(pdf(gm,x));
end
%% define PMC sampler
test_case = 6;

if test_case == 0 % original class, each population just 1 sample
    N = 1000;
    mu0 = mvnrnd([0,0],10*[1,0;0,1],N);
    pmc = PMCPlain(mu0,logTarget);
end
if any(test_case == [1,4]) % each population just 1 sample
    N = 1000;
    mu0 = mvnrnd([0,0],10*[1,0;0,1],N);
    pmc = PMC(logTarget,mu0,1);
end
if any(test_case == [2,5]) % global resampling (default resdampling)
    N = 30;
    mu0 = mvnrnd([0,0],10*[1,0;0,1],N);
    pmc = PMC(logTarget,mu0,30);
end
if any(test_case == [3,6]) % local resampling
    N = 30;
    mu0 = mvnrnd([0,0],10*[1,0;0,1],N);
    pmc = PMC(logTarget,mu0,30);
    pmc.resample_method = 'local';
end

% pmc.dmw = 0;

%% sampling cycles
I = 50;

if any(test_case == [1,2,3])
    % fixed covariance and temperature
    pmc.setSigma(10);
    pmc.setTemp(0.1)
    for i = 1:I
        pmc.sample()
        if pmc.D == 2
            plot2dPost(pmc)
        end
        summary(pmc, pdf_case)
    end
end

if any(test_case == [4,5,6])
    % changing covariance and temperature
    T = logspace(-2,0,I);
    S = logspace(2,0,I);
    for i = 1:I
        pmc.setTemp(T(i))
        pmc.setSigma(S(i))
        pmc.sample()
        if pmc.D == 2
            plot2dPost(pmc)
        end
        summary(pmc, pdf_case)
    end
end

%% visulization function
function summary(pmc, pdf_case)
clc
[x_p, w_p] = pmc.posterior();
if pdf_case == 1
    [mu_c,C_c] = mvnfit(x_p,w_p)
elseif pdf_case == 2
    x_p = resample(x_p, w_p) + randn(size(x_p)) * 0.1;
    gm_estimate = fitgmdist(x_p,5);
    gm_estimate.ComponentProportion
    gm_estimate.mu
    gm_estimate.Sigma(:,:,1)
end
end

function plot2dPost(pmc)
[x_p, w_p] = pmc.posterior();
x = resample(x_p, w_p) + randn(size(x_p)) * 0.1;
plot2Dsample(x,pmc.logTarget)
end

function plot2Dsample(x,logTarget)
pause(0.001)
hold off
plot(x(:,1),x(:,2),'x')
hold on
if nargin == 2
    plotContour(logTarget,x)
else
end
% axis([-20,16,-20,16])
end

function plotContour(t,x)
maxr = max(x);
minr = min(x);

x = linspace(minr(1),maxr(1));
y = linspace(minr(2),maxr(2));
[X,Y] = meshgrid(x,y);
Z = reshape(exp(t([X(:),Y(:)])),size(X));
contour(X,Y,Z)
end