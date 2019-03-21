classdef PMC < handle
    % Plain PMC based on algorithm 1
    properties
        % model definition
        logTarget % N*D-> N*1: target distribution
        
        % PMC parameters fixed
        N % number of populations
        K % number of samples per population
        D % dimension of sample
        
        % PMC parameter mutable
        mu % proposal means
        C % proposal covariance
        resample_method % flag choosing resample method
        
        
        % PMC sample records
        data % cell array of Ppdf objects
    end
    
    methods
        function O = PMC(logTarget, mu0, K)
            % initialization function
            % x0: N*D: initial sample of the states
            % logTarget: function handle: log target function54
            
            % pass variables
            O.logTarget = logTarget;
            O.mu = mu0;
            [O.N, O.D] = size(mu0);
            O.K = K;
            
            % default parameters
            O.C = eye(O.D);
            O.resample_method = 'global';
            
            % initial data array,
            O.data = cell(O.N, 0);
        end
        function setSigma(O,sig)
            % set the proposal covariance by the
            O.C = eye(O.D)*(sig^2);
        end
        function sample(O)
            % perform one iteration of sampling
            I = size(O.data,2);
            for n = 1:O.N
                x_n = zeros([O.K,O.D]); % samples for current population
                logw_n = zeros([O.K,1]); % log weights of samples for current population
                for k = 1:O.K
                    % sample and compute log weights
                    x_n(k,:) = mvnrnd(O.mu(n,:),O.C);
                    logw_n(k) = O.logTarget(x_n(k,:)) - logmvnpdf(x_n(k,:),O.mu(n,:),O.C);
                end
                % record samples and unnormalized weights
                O.data{n,I+1} = Ppdf(x_n, logw_n);
            end
            % use normalized weights for resampling
            if strcmp(O.resample_method, 'global')
                O.mu = Ppdf.globalResample(O.data(:,end));
            elseif strcmp(O.resample_method, 'local')
                O.mu = Ppdf.localResample(O.data(:,end));
            else
                error(['resample must one of the following',newline,'"global", "local"'])
            end
        end
        function [x_p, w_p] = posterior(O)
            % reshape the data for a better representation of posterior.
            data_p = Ppdf.merge(O.data);
            x_p = data_p.x;
            w_p = logw2w(data_p.logw);
        end
    end
end
