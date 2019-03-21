% 2D example showing the difference of tempering and convolution
clear
close all
clc

sig = 0.1;

target = @(x) normpdf(x,1,sig) + normpdf(x,-1,sig);

t = -5:0.01:5;

if 0
    zeta = 0.9;
    lambda = 1;
    for i = 1:100
        pause(0.1)
        subplot(2,1,1)
        plot(t, target(t).^lambda)
        title('unzoomed')
        axis([-5,5,0,1])
        
        subplot(2,1,2)
        plot(t, target(t).^lambda)
        title('zoomed in')
        
        lambda = lambda * zeta;
    end
else
    M = 1000;
    J = 20;
    sig_noise = 5;
    zeta = 5^(-1/J);
    w = zeros(size(t));
    for j = 1:J
        meth_conv = 1;
        if meth_conv == 1
            u = randn([1,M]) * sig_noise;
            for i = 1:length(t)
                w(i) = mean(target(t(i) + u));
            end
        end
        if meth_conv == 2
            w = target(t + randn(size(t)) * sig_noise);
        end
        
        t_ = t;
        w_ = w;
        
        pause(0.5)
        subplot(3,1,1)
        plot(t_, w_, '.')
        title('unzoomed')
        axis([-5,5,0,1])
        
        subplot(3,1,2)
        plot(t_, w_, '.')
        title('zoomed in')
        
        subplot(3,1,3)
        Plot1dParticleCDF(t_,w_)
        title('cdf')
        
        sig_noise = sig_noise * zeta;
    end
end

function Plot1dParticleCDF(x,w)
w = w(:)/sum(w(:));
x= x(:);
data = [x,w];
data = sortrows(data,1);
x = data(:,1);
w = data(:,2);
plot(x,cumsum(w));
end