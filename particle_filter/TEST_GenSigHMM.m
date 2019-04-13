% Example of GenSig object usage
%% clean up
clear
close all
clc

%% defining the object
T = 10; % length of the signal
G = GenSig(T, @RndTr, @RndOb); % create a generator function

%% usage of the generator function
% here, we just plot the datapoints to test the funciton.
% however, in real use, people woulf like to use Bayesian filters for
% sequential estimations.

% testing yeild observations
figure
subplot(4, 1, 1)
title('sequential yielded observations y')
hold on
while G.hasnext
    plot(G.n, G.next, 'o') % G.n is not usually used in estimation, but for plot only.
    pause(0.1) % for animation
end

% testing yeild the same observations again
G.restart % reset the generator
while G.hasnext
    % same cycle but different legend
    plot(G.n, G.next, 'x') 
    pause(0.1)
end

%% direct plot of the data
% drawing the data is usually used in the performance evaluation of state
% estimations.

% plot observations
subplot(4, 1, 2)
title('direct read observation data y')
hold on
plot(G.y)

% plot the states
subplot(4, 1, 3)
title('direct read state data x')
hold on
plot(G.x)

%% Generate a new signal
G = GenSig(T, @RndTr, @RndOb); % a new signal is generated by create a new object

subplot(4, 1, 4)
title('another generator object')
hold on
while G.hasnext
    % same cycle but different legend
    plot(G.n, G.next, '+') 
end
plot(G.y)


%% model functions
function x = RndTr(x,randomOn)
% state ransition function
if nargin == 1
    randomOn = 1;
end
mux = -1;
a = 0.9 ;
sigmax = 0.1;
x = mux + a*(x - mux) + sigmax*randn(size(x))*randomOn;
end
function y = RndOb(x)
% observation function
y = exp(x/2) .* randn(size(x));
end