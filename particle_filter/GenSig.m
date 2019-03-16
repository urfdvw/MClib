classdef GenSig < handle % to avoid deep copy of data in each time step
    % signal generator function given a State Space Model (SSM)
    properties
        % model and its parameters
        T % number of data points needed
        RndTr % state transition process
        RndOb % observation process
        
        % data
        x % states
        y % observations
        
        % generator related
        n % time index
    end
    methods
        function O = GenSig(T, RndTr, RndOb)
            % construction function
            
            % variable pass
            O.T = T;
            O.RndTr = RndTr;
            O.RndOb = RndOb;
            
            % extra parameters for datageneration
            Tstart = 100; % extra burn-in steps added before beginning to stablize the process
            x0 = randn; % initial state, gaussian random
            
            % generate the states
            x = zeros(T + Tstart, 1); % memory allocation
            x(1) = O.RndTr(x0); % step t = 1
            for t = 2 : T+Tstart
                % steps t > 1
                x(t) = O.RndTr(x(t - 1));
            end
            O.x = x(Tstart+1 : end); % only the last T steps are recorded
            O.y = O.RndOb(O.x); % observation steps
            restart(O); % initialize time index
        end
        function result = hasnext(O)
            % return true if more data are available, o.w. false
            % (minicing a generator fucntion)
            result = O.n <= O.T;
        end
        function y = next(O)
            % yeild the observation of the next time step
            % (mimicing a generator function)
            y = O.y(O.n); % get observation data
            O.n = O.n+1; % move the time index
        end
        function restart(O)
            % restart the generator from the beginning
            O.n = 1; % set the time index to the beginning
        end      
    end
end