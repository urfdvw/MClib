% test Ppdf
clear
close all
clc
addpath('..\functions\')

M = 100;
D = 2;
u = randn([M,D]);
logw = log(mvnpdf(u));

l = Ppdf(u,logw);
A = {l,l;l,l};
Ca = Ppdf.merge(A);