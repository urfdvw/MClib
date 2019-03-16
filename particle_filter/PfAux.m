classdef PfAux < handle
    properties
        % model definition
        RndTr
        LiOb
        % Pf parameters
        M
        % Pf samples
        x
        w
        x_
        w_
    end
    
    methods
        function O = PfAux(x0,RndTr,LiOb)
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
        end
        function xHat = estimate(O,y)
            O.x_ = O.RndTr(O.x,0);
            O.w_ = O.LiOb(y,O.x_);
            km = IS('sample_n',O.M,'weights',O.w.*O.w_);
            O.x = O.RndTr(O.x(km,:)); % propose by transition 
            O.w = O.LiOb(y,O.x)./O.w_(km); % likelihood of states by observation
            O.w = O.w / sum(O.w); % normalize the states
            xHat = O.w' *O.x; % make estimation by weighted sum
        end
    end
end

