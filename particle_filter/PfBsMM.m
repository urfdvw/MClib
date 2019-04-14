classdef PfBsMM < handle
    % Multi transition model boot strap particle filter
    properties
        % model definition
        % Transition process group
        RndTrs % cell array vector, function handle, for each one of the M states.
            % Input:
                % x: M*Dx matrix, M: number of particles, Dx: dimension of states
                % random on: bool: choose the usage of the function
                    % 1: sample state particles from the transition model
                    % 0: noiseless prediction of states by transition model
            % Output:
                % x: same size and meaning as input x
                
        % Observation Likelihood
        LiOb % function handle, for each one of the M states.
            % Input:
                % y: 1*Dy, Dy: dimension of observation
                % x: M*Dx matrix, M: number of particles, Dx: dimension of states
            % Output:
                % p: M*1, likelihood of each state
                
        % Pf size
        D % dimension of states
        M % scaler, number of particles per-model
        N % number of models
        
        % Pf samples
        particles
        
        % Model likelihood
        MW % N*1 vector contains the model weight
        
        % Pf parameters
        rho % scaler, the forgetting parameter
    end
    
    methods
        function O = PfBsMM(x0,RndTrs,LiOb)% construction by model
            
            % x0: M*D: initial sample of the states
            % RndTr: cell array vectot: function handle: transition function of the states
            % LiOb: function handle: likelihood of the states based on observation function
            
            % pass variables
            [O.M, O.D] = size(x0);
            RndTrs = RndTrs(:);
            O.RndTrs = RndTrs;
            O.N = length(RndTrs);
            O.LiOb = LiOb;
            % all models have same initial particles
            O.particles = cell([O.N, 1]);
            for n = 1: O.N
                O.particles{n} = modelParticles(x0, ones([O.M, 1])/O.M);
            end
            % initial value
            O.MW = ones([O.N, 1]);
            
            % default parameter
            O.rho = 0;
        end
        function xHat= estimate(O,y)
            % for each model:
            ML = zeros([O.N, 1]);
            for n = 1: O.N
                % propose particles of each model
                O.particles{n}.x = O.RndTrs{n}(O.particles{n}.x);
                % calculate pricate weight of each model
                O.particles{n}.w = O.LiOb(y, O.particles{n}.x);
                % calculate the model likelihoods
                ML(n) = sum(O.particles{n}.w);
            end
            % update model weights by model likelihoods
            O.MW = O.MW/sum(O.MW);
            O.MW = O.MW.^O.rho .* ML;
            % down sample the particle by global weight
            xHat = modelParticles.globalResample(O.particles, O.MW);
        end
    end
end

