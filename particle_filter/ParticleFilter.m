classdef ParticleFilter < handle
    % particle filter base class
    properties
        % model definition
        % Transition process
        RndTr % function handle, for each one of the M states.
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
                
        % Pf parameters
        M % scaler, number of particles
        
        % Pf samples
        x % M*Dx, particle position
        w % M*1 particle weights
        
        % Model likelihood
        ML % scaler
    end
    
    methods
        function O = ParticleFilter(x0,RndTr,LiOb)
            % construction by model
            
            % x0: M*D: initial sample of the states
            % RndTr: function handle: transition function of the states
            % LiOb: function handle: likelihood of the states based on observation function
            
            % pass variables
            O.M = size(x0,1);
            O.x = x0;
            O.RndTr = RndTr;
            O.LiOb = LiOb;
            % initial value
            O.w = ones(O.M,1)/O.M;
            O.ML = 1;
        end
        function continue_from(O, ML, x, w)
            % construction by other particle filters
            
            % ML: scaler, model likelihood
            % x: M*Dx: acquired particle positions
            % w: M*1 : acquired particle weights
            
            O.ML = ML;
            O.M = size(x,1);
            O.x = x;
            if nargin == 3
                O.w = ones(O.M,1)/O.M;
            else
                O.w = w;
            end
        end
    end
    
    methods (Abstract)
        [xHat, ML] = estimate(O,y) % estimation functino
    end
end

