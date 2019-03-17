classdef PMCPlain < handle
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
            % x0: N*D: initial sample of the states
            % logTarget: function handle: log target function54
            
            % pass variables
            O.logTarget = logTarget;
            O.mu = x0;
            [O.N,O.D] = size(x0);
            O.x = zeros(O.N,O.D,0);
            O.w = zeros(O.N,0);
            
            % default parameters
            O.C = 0.1 * eye(O.D);
        end
        function setSigma(O,sig)
            O.C = eye(O.D)*(sig^2);
        end
        function sample(O,I)
            if nargin == 1
                I = 1;
            end
            x_append = zeros([O.N,O.D,I]);
            w_append = zeros([O.N,I]);
            for i = 1:I
                x_c = zeros([O.N,O.D]);
                logx_c = zeros(O.N,1);
                for n = 1:O.N
                    x_c(n,:) = mvnrnd(O.mu(n,:),O.C);
                    logx_c(n) = O.logTarget(x_c(n,:)) - logmvnpdf(x_c(n,:),O.mu(n,:),O.C);
                end
                w_append(:,i) = exp(logx_c);
                x_append(:,:,i) = x_c;
                w_c = logw2w(logx_c);
                O.mu = resample(x_c,w_c);
            end
            O.x = cat(3,O.x,x_append);
            O.w = cat(2,O.w,w_append);
        end
        function x_hat = estimate(O)
            w_c = O.w/sum(O.w(:));
            x_hat = zeros([1,O.D,1]);
            for n = 1:O.N
                for i = 1:size(O.x,3)
                    x_hat = x_hat + w_c(n,i) * O.x(n,:,i);
                end
            end
        end
    end
end
