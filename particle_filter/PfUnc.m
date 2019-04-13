classdef PfUnc < ParticleFilter
    % Uncented Particle Filter
    properties
    end
    
    methods
        function O = PfUnc(x0,RndTr,LiOb)
            O = O@ParticleFilter(x0,RndTr,LiOb);
        end
        function [xHat, ML] = estimate(O,y)
            O.x = O.RndTr(O.x); % propose by transition 
            O.w = O.LiOb(y,O.x); % likelihood of states by observation
            ML = sum(O.w); % compute the model likelihood
            O.ML = ML; % log the result
            O.w = O.w / sum(O.w); % normalize the weights
            [mu,Sigma]=mvnfit(O.x,O.w); % approximate the weighted samples by Gaussian distribution
            xHat = mu; % make estimation by the mu
            O.x = SigmaPoints(mu,Sigma); % use sigma points as samples
            O.w = ones(O.M,1)/O.M; % even weights for the new samples
        end
    end
end

