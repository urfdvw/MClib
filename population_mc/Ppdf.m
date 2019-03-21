classdef Ppdf
    
    properties
        % data
        x
        logw
        % parameters
        D
    end
    
    methods
        function O = Ppdf(x,logw)
            O.x = x;
            O.D = size(x,2);
            if nargin == 1
                O.logw = zeros(1,O.D);
            else
                O.logw = logw;
            end
        end
        
        function O = append(O1, O2)
            O = Ppdf([O1.x;O2.x],[O1.logw;O2.logw]);
        end
    end
    
    methods(Static)
        function O = merge(OCA)
            % OCA: cell array of SUCH object
            OCA = OCA(:);
            O = OCA{1};
            for i = 2:length(OCA)
                O = O.append(OCA{i});
            end
        end
        
        function mu = globalResample(OCA)
            N = size(OCA, 1);
            O = Ppdf.merge(OCA);
            ind = indexSample(N, logw2w(O.logw));
            mu = O.x(ind,:);
        end
        
        function mu = localResample(OCA)
            [M,N] = size(OCA);
            D = OCA{1}.D; 
            mu = zeros([0,D]);
            i = 1;
            for m = 1:M
                for n = 1:N
                    O = OCA{m,n};
                    ind = indexSample(1, logw2w(O.logw));
                    mu(i,:) = O.x(ind,:);
                    i = i + 1;
                end
            end
        end
    end
end

