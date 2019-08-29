clear
close all
clc

%% model functions
rnd_terran = {}; % floor model
rnd_roof = {}; % floor model
% % x_t = x_{t-1}
I = 1; % model counter
rnd_terran{I} = @(x_) x_; 
rnd_roof{I} = @(x_) x_; 
% % x_t = x_{t-1} + eps 
I = I + 1;
sig = 0.1;
rnd_terran{I} = @(x_) x_ + sig*randn(size(x_)); 
rnd_roof{I} = @(x_) x_ + sig*randn(size(x_)); 
% % x_t = C
H = 10; % height of roof
I = I + 1;
rnd_terran{I} = @(x_) zeros(size(x_)); 
rnd_roof{I} = @(x_) H*ones(size(x_)); 
% % x_t = C + esp, esp \sim \mathcal{N}
I = I + 1;
sig = 1;
rnd_terran{I} = @(x_) sig*randn(size(x_)); 
rnd_roof{I} = @(x_) H + sig*randn(size(x_)); 
% % x_t = C +- esp, esp \sim \mathcal{exp}
I = I + 1;
mu = 1;
rnd_terran{I} = @(x_) exprnd(mu, size(x_)); 
rnd_roof{I} = @(x_) H - exprnd(mu, size(x_)); 
% drone model
sig = 0.2;
rnd_drone = @(x_) x_ + sig*randn(size(x_));
model = {};
N = 1;
for it = 1 : I
    for ir = 1 : I
        model{N} = @(x) [rnd_drone(x(:,1)), ...
                         rnd_roof{ir}(x(:,2)), ...
                         rnd_terran{it}(x(:,3))];
        N = N + 1;
    end
end

model{23}(ones(100,3))