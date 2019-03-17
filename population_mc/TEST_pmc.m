% test PMC algorithms
clear
close all
clc

logTarget = @(x)log(mvnpdf(x,[0,0],[1,0;0,1]));
N = 1000;
x0 = mvnrnd([0,0],10*[1,0;0,1],N);
plot2Dsample(x0)
pmc = PMCPlain(x0,logTarget);
pmc.setSigma(0.01);
for i = 1:10
    pmc.sample(100)
    plot2Dsample(pmc.mu)
    x = reshape(pmc.x,[],2);
    w = pmc.w(:);
    [mu_c,C_c] = mvnfit(x,w)
end
function plot2Dsample(x)
pause(0.1)
plot(x(:,1),x(:,2),'.')
axis([-10,10,-10,10])
end