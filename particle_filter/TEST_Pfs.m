clear
close all
clc
addpath('..\functions\')
%% Parameters
M = 1000; % number of samples
T = 400; % time steps for estimate
Terror = 200; % times steps for error calculation, after burn-in
%% signal generator
signal = GenSig(T,@RndTr,@RndOb);
%% Choose a filter by uncomment 

filter = PfBs(randn(M,1),@RndTr,@LiOb);
% filter = PfGau(randn(M,1),@RndTr,@LiOb);
% filter = PfUnc(randn(M,1),@RndTr,@LiOb);
% filter = PfAux(randn(M,1),@RndTr,@LiOb);

%% filter cycle

xHat = zeros(T,1); % memory allocation
t = 1; % initialize time index
signal.restart % read signal from start
while signal.hasnext % if there are more signals to go
    % estimation cycle
    xHat(t) = filter.estimate(signal.next); % estimate by new coming observation
    t = t+1; % update time index
end

%% result
diff = xHat - signal.x;
error = rms(diff(T-Terror:T));
hold on
plot(signal.x)
plot(xHat,'o')

%% model functions
function x = RndTr(x,randomOn)
% state ransition function
if nargin == 1
    randomOn = 1;
end
mux = 10;
a = 0.8 ;
sigmax = 0.5;
x = mux + a*(x - mux) + sigmax * randn(size(x))*randomOn;
end
function y = RndOb(x)
% observation function
y = exp(x/2) .* randn(size(x));
end
function p = LiOb(y,x)
% likelihood
p = normpdf(y,y*0,exp(x/2));
end