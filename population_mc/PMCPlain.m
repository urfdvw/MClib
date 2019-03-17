classdef PMCPlain < handle
    % Plain PMC based on algorithm 1
    properties
        % model definition
        logTarget % N*D-> N*1: target distribution
        % PMC parameters
        N % number of samples
        D % dimension of sample
        mu % proposal means
        C % proposal covariance
        % PMC sample records
        x % N*D*I
        w % N*I
    end
    
    methods
        function O = PMCPlain(x0,logTarget)
            % initialization function
            % x0: N*D: initial sample of the states
            % logTarget: function handle: log target function54
            
            % pass variables
            O.logTarget = logTarget;
            O.mu = x0;
            [O.N,O.D] = size(x0);
            O.x = zeros(O.N,O.D,0);
            O.w = zeros(O.N,0);
            
            % default parameters
            O.C = eye(O.D);
        end
        function setSigma(O,sig)
            % set the proposal covariance by the 
            O.C = eye(O.D)*(sig^2);
        end
        function sample(O,I)
            % perform I iterations of sampling
            if nargin == 1
                % if number of iterations not given, just perform 1
                I = 1;
            end
            % alocate space for new resulte
            x_append = zeros([O.N,O.D,I]);
            w_append = zeros([O.N,I]);
            % main cycle
            for i = 1:I
                x_c = zeros([O.N,O.D]); % samples for current cycle
                logw_c = zeros(O.N,1); % log weights of samples for current cycle
                for n = 1:O.N
                    % sample and compute log weights
                    x_c(n,:) = mvnrnd(O.mu(n,:),O.C);
                    logw_c(n) = O.logTarget(x_c(n,:)) - logmvnpdf(x_c(n,:),O.mu(n,:),O.C);
                end
                % record samples and unnormalized weights
                w_append(:,i) = exp(logw_c);
                x_append(:,:,i) = x_c;
                % use normalized weights for resampling
                w_c = logw2w(logw_c);
                O.mu = resample(x_c,w_c);
            end
            % add new results to embedded data
            O.x = cat(3,O.x,x_append);
            O.w = cat(2,O.w,w_append);
        end
        function [x_p, w_p] = posterior(O)
            % reshape the data for a better representation of posterior.
            I = size(O.x,3);
            x_p = zeros([O.N*I,O.D]);
            w_p = zeros([O.N*I,1]);
            w_c = O.w/sum(O.w(:));
            ip = 1;
            for n = 1:O.N
                for i = 1:size(O.x,3)
                    w_p(ip) = w_c(n,i);
                    x_p(ip,:) = reshape(O.x(n,:,i),[1,O.D]);
                    ip = ip + 1;
                end
            end
        end
    end
end
