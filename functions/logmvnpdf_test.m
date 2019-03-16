% test different versions of logmvnpdf() and measure the speed
clear
close all
clc

%% data
sig = 1;

D=100;
N=10000;
mu=randn(D,1);
sig=eye(D) * sig;
x=randn(D,N);


%% build in
tic
p1=log(mvnpdf(x',mu',sig));
toc
%% old version
tic
p2=logmvnpdf_old_version(x,mu,sig);
toc
%% bsf version
tic
p3=logmvnpdf(x,mu,sig);
toc