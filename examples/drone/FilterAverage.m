classdef FilterAverage < handle
    
    properties
        % filter parameter
        Threshold
        % filter stored data
        last_height
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
            O.Threshold = 1;
        end
        
        function O = setThreshold(O,th)
            O.Threshold = th;
        end
        
        function height = estimate(O,data)
            if O.just_started
                height = data;
                O.just_started = 0;
            else
                height = data + O.displacement;
                displacement_c = height - O.last_height;
                if abs(displacement_c) > O.Threshold
                    O.displacement = O.displacement - displacement_c;
                    height = data + O.displacement;
                end
            end
            O.last_height = height;
        end
    end
end

