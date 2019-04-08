classdef PfBs < handle
    properties
        % model definition
        RndTr
        LiOb
        % Pf parameters
        M
        % Pf samples
        x
        w
        % Model likelihood
        ML
    end
    
    methods
        function O = PfBs(x0,RndTr,LiOb)
            % x0: M*D: initial sample of the states
            % RndTr: function handle: transition function of the states
            % LiOb: function handle: likelihood of the states based on
            % observation function
            
            % pass variables
            O.M = size(x0,1);
            O.x = x0;
            O.w = ones(O.M,1)/O.M;
            O.RndTr = RndTr;
            O.LiOb = LiOb;
            O.ML = 1;
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
            O.w = O.w / ML; % normalize the states
            xHat = O.w' *O.x; % make estimation by weighted sum
            O.ML = ML;
        end
        function continue_from(O, x, w)
            O.M = size(x,1);
            O.x = x;
            if nargin == 2
                O.w = ones(O.M,1)/O.M;
            else
                O.w = w;
            end
        end
    end
end

