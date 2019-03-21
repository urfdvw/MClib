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
        lambda % inverse temperture of likelihood
        dmw % flag of using DM-weight
        
        
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
            O.lambda = 1;
            O.resample_method = 'global';
            O.dmw = 1;
            
            % initial data array,
            O.data = cell(O.N, 0);
        end
        function setSigma(O,sig)
            % set the proposal covariance by the
            O.C = eye(O.D)*(sig^2);
        end
        function setTemp(O,lambda)
            O.lambda = lambda;
        end
        function sample(O)
            % perform one iteration of sampling
            I = size(O.data,2);
            data_ip1 = cell(O.N,1);
            data_temp = cell(O.N,1);
            parfor n = 1:O.N
                x_n = zeros([O.K,O.D]); % samples for current population
                logw_n = zeros([O.K,1]); % log weights of samples for current population
                logTw_n = zeros([O.K,1]); % log tempered weights of samples for current population
                for k = 1:O.K
                    % sample and compute log weights
                    x_n(k,:) = mvnrnd(O.mu(n,:),O.C);
                    if O.dmw
                        logprop =  logmean(logmvnpdf(O.mu,x_n(k,:),O.C));
                    else
                        logprop = logmvnpdf(x_n(k,:),O.mu(n,:),O.C);
                    end
                    logw_n(k) = O.logTarget(x_n(k,:)) - logprop;
                    logTw_n(k) = O.lambda * O.logTarget(x_n(k,:)) - logprop;
                end
                % record samples and unnormalized weights
                data_ip1{n} = Ppdf(x_n, logw_n);
                data_temp{n} = Ppdf(x_n, logTw_n);
            end
            % attach untempered unnormalized data to posterior
            O.data = [O.data,data_ip1];
            % use tempered normalized weights for resampling
            if strcmp(O.resample_method, 'global')
                O.mu = Ppdf.globalResample(data_temp);
            elseif strcmp(O.resample_method, 'local')
                O.mu = Ppdf.localResample(data_temp);
            else
                error(['resample must one of the following',newline,'"global", "local"'])
            end
        end
        function sample_cycles(O)
        end
        function [x_p, w_p] = posterior(O)
            % reshape the data for a better representation of posterior.
            L = max(1, size(O.data,2)-10);
            data_p = Ppdf.merge(O.data(:,L:end));
            x_p = data_p.x;
            w_p = logw2w(data_p.logw);
        end
    end
end
