classdef modelParticles < handle
    % particles of one model
    properties
        % data
        x
        w
        % parameters
        M
        D
    end
    
    methods
        function O = modelParticles(x,w)
            O.x = x;
            [O.M, O.D] = size(x);
            if nargin == 1
                O.w = ones(1,O.D);
            else
                O.w = w;
            end
        end
    end
    
    methods(Static)
        function xhat = globalResample(OCA, MW)
            % acquire global particles by a cell of such object
            % and reweight the particles by model weights
            % and do resampling
            
            % OCA: cell array of SUCH object
            % MW: model weights
            
            % pass data
            OCA = OCA(:); % reshape cell
            M = OCA{1}.M;
            D = OCA{1}.D;
            
            % get all particles and reweight
            x = zeros([0, D]);
            w = zeros([0, 1]);
            for i = 1: length(MW)
                x = [x; OCA{i}.x];
                w_temp = OCA{i}.w;
                w = [w; w_temp/sum(w_temp(:))*MW(i)];
            end
            
            % estimation
            xhat = weighted_mean_nan(x,w);
            
            % global resample
            ind = indexSample(M, w);
            for i = 1: length(MW)
                OCA{i}.x = x(ind,:);
                OCA{i}.w = ones([M,1])/M;
            end
        end       
    end
end

