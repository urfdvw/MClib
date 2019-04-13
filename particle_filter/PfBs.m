classdef PfBs < ParticleFilter
    % boot strap particle filter
    properties
    end
    
    methods
        function O = PfBs(x0,RndTr,LiOb)
            O = O@ParticleFilter(x0,RndTr,LiOb);
        end
        function [xHat, ML] = estimate(O,y)
            O.x = resample(O.x,O.w); % resample the particles
            % if ESS(O.w) < 0.5
            %     O.x = resample(O.x,O.w); % resample the particles
            %     % O.w = ones(O.M,1)/O.M; % does not have effect
            % end
            O.x = O.RndTr(O.x); % propose by transition
            O.w = O.LiOb(y,O.x); % likelihood of states by observation
            ML = sum(O.w); % compute the model likelihood
            O.ML = ML; % log the result
            O.w = O.w / ML; % normalize the weights
            xHat = O.w' *O.x; % make estimation by weighted sum
        end
    end
end

