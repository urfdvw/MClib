classdef PfAux < ParticleFilter
    % Auxiliary Particle Filter
    properties
        % Auxiliary prediction streams
        x_
        w_
    end
    
    methods
        function O = PfAux(x0,RndTr,LiOb)
            O = O@ParticleFilter(x0,RndTr,LiOb);
        end
        function [xHat, ML] = estimate(O,y)
            O.x_ = O.RndTr(O.x,0); % noiseless states predictions
            O.w_ = O.LiOb(y,O.x_); % likelihood of the predictions
            km = indexSample(O.M,O.w.*O.w_); % sample the streams by their prediction performance
            O.x = O.RndTr(O.x(km,:)); % propose by transition 
            O.w = O.LiOb(y,O.x)./O.w_(km); % likelihood of states by observation
            ML = sum(O.w); % compute the model likelihood
            O.ML = ML; % log the result
            O.w = O.w / sum(O.w); % normalize the states
            xHat = O.w' *O.x; % make estimation by weighted sum
        end
    end
end

