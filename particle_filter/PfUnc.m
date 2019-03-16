classdef PfUnc < handle
    properties
        % model definition
        RndTr
        LiOb
        % Pf parameters
        M
        % Pf samples
        x
        w
    end
    
    methods
        function O = PfUnc(x0,RndTr,LiOb)
            % x0: M*D: initial sample of the states
            % RndTr: function handle: transition function of the states
            % LiOb: function handle: likelihood of the states based on
            % observation function
            
            % pass variables
            O.M = size(x0,2)*2;
            O.x = x0;
            O.w = ones(O.M,1)/O.M;
            O.RndTr = RndTr;
            O.LiOb = LiOb;
        end
        function xHat = estimate(O,y)
            O.x = O.RndTr(O.x); % propose by transition 
            O.w = O.LiOb(y,O.x); % likelihood of states by observation
            O.w = O.w / sum(O.w); % normalize the states
            [mu,Sigma]=mvnfit(O.x,O.w); % use MVNormal distribution to approximate
            xHat = mu; % make estimation by the mu
            O.x = SigmaPoints(mu,Sigma); % sample from approximated Gaussian
            O.w = ones(O.M,1)/O.M; % does not have effect
        end
    end
end

