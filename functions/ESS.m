function out = ESS(W)
% Efective sample size
out=1/sum((W(~isinf(W))).^2)/length(W);
end

