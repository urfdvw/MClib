classdef FilterAverage < handle
    
    properties
        % filter parameter
        threshold
        constant_value
        % filter stored data
        last_data
        displacement
        % initialization flag
        just_started
    end
    
    methods
        function O = FilterAverage()
            % initialize data
            O.just_started = 1;
            O.displacement = 0;
            % default parameter
            O.threshold = 1;
            O.constant_value = 0.1;
        end
        
        function O = set_threshold(O,th)
            O.threshold = th;
        end
        
        function O = set_correctionC(O,C)
            O.constant_value = C;
        end
        
        function height = estimate(O,data)
            if O.just_started
                height = data;
                O.just_started = 0;
            else
                if abs(O.last_data-data) > O.threshold
                    O.displacement = O.displacement + (O.last_data - data);
                end
                height = data + O.displacement;
                if O.displacement > 0
                    O.displacement = O.displacement - O.constant_value;
                elseif O.displacement < 0
                    O.displacement = O.displacement + O.constant_value;
                end
            end
            O.last_data = data;
        end
    end
end

