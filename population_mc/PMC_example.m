% example of PMC class usage
%% clean up set path
clear
close all
clc
addpath('..\functions\')

%% 2D Gaussian mixture target distribution
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

%% define PMC sampler
N = 30; % number of particles per population
K = 50; % number of populations
mu0 = mvnrnd(zeros([1,ND]), 3*eye(ND),N); % initial mean of each population
pmc = PMC(logTarget,mu0,K); % define pmc object
pmc.resample_method = 'local'; % set to local resampling
disp(pmc.info) % print out current settings
%% sampling cycles
I = 50;
figure
T = logspace(-2,0,I); % plan temperature changing
S = logspace(2,0,I); % plan covariance changing
for i = 1:I
    pmc.setTemp(T(i)) % tempering the target
    pmc.setSigma(S(i)) % set proposal width
    pmc.sample() % main operation
    plot2dPost(pmc) % plot the result
end

%% visulization function
function plot2dPost(pmc)
if 1 % 1: plot means, 0: plot samples
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
axis([-15,18,-20,20])
end

function plotContour(t,x)
maxr = max(x) + 4;
minr = min(x) - 4;

x = linspace(minr(1),maxr(1));
y = linspace(minr(2),maxr(2));
[X,Y] = meshgrid(x,y);
Z = reshape(exp(t([X(:),Y(:)])),size(X));
contour(X,Y,Z)
end