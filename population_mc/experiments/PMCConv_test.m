% test PMC algorithms
clear
close all
clc
addpath('..\functions\')

%% target distribution
pdf_case = 4;

if pdf_case == 1
    ND = 2;
    logTarget = @(x)logmvnpdf(x,[-10,-10],[2,0.6;0.6,1]);
elseif pdf_case == 2
    ND = 2;
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
elseif pdf_case == 3
    ND = 30;
    mu = -10 * ones([1,ND]);
    sigma = 0.1 * eye(ND);
    logTarget = @(x)logmvnpdf(x,mu,sigma);
elseif pdf_case == 4
    ND = 2;
    mu = -10 * ones([1,ND]);
    sigma = 0.000001 * eye(ND);
    logTarget = @(x)logHDTarget(x,mu,sigma);
end
%% define PMC sampler

N = 30;
mu0 = mvnrnd(10*ones([1,ND]), 3*eye(ND),N);
if 1
    pmc = PMC(logTarget,mu0,30);
else
    pmc = PMCConv(logTarget,mu0,30);
end
pmc.resample_method = 'local';


% pmc.dmw = 0;

%% sampling cycles
I = 50;

% changing covariance and temperature
T = logspace(-10,0,I);
S = logspace(1,0,I);
for i = 1:I
    pmc.setTemp(T(i))
    pmc.setSigma(S(i))
    pmc.sample()
    if pmc.D == 2
        plot2dPost(pmc)
%         hold on
%         plotContour(@(x)target_conv(pmc.logTarget,pmc.sig_conv,x),pmc.mu)
%         pause(0.001)
    end
    try
        summary(pmc, pdf_case)
    catch
    end
end

%% visulization function
function summary(pmc, pdf_case)
[x_p, w_p] = pmc.posterior();
if any(pdf_case == [1,3])
    clc
    [mu_c,C_c] = mvnfit(x_p,w_p)
elseif any(pdf_case == [2,4])
    x_p = resample(x_p, w_p) + randn(size(x_p)) * 0.1;
    if pdf_case == 2
        gm_estimate = fitgmdist(x_p,5);
    elseif pdf_case == 4
        gm_estimate = fitgmdist(x_p,2);
    end
    clc
    gm_estimate.ComponentProportion
    gm_estimate.mu
    %     gm_estimate.Sigma(:,:,1)
end
end

function plot2dPost(pmc)
if 1
    x = pmc.mu;
else
    [x_p, w_p] = pmc.posterior();
    x = resample(x_p, w_p) + randn(size(x_p)) * 0.1;
end
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
N = 100;

maxr = max(x) + 3;
minr = min(x) - 3;

x = linspace(minr(1),maxr(1),N);
y = linspace(minr(2),maxr(2),N);
[X,Y] = meshgrid(x,y);
Z = reshape(t([X(:),Y(:)]),size(X));
contour(X,Y,Z)
end