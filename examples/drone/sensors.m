classdef sensors < handle
    
    properties
        roof % function handle
        terran % function handle
        path % N*2 matrix
        
        data % a tructure storing all the semsor data
        
        i % index
    end
    
    methods
        function O = sensors(roof,terran,path)
            O.roof = roof;
            O.terran = terran;
            O.path = path;
            
            N = size(path,1);
            
            O.data.infraredup = zeros(N,1);
            O.data.infrareddown = zeros(N,1);
            O.data.ultrasound = zeros(N,1);
            
            for n = 1:N
                O.data.infraredup(n) = O.roof(O.path(n,1)) - O.path(n,2) + randn * 0.1;
                O.data.infrareddown(n) = O.path(n,2) - O.terran(O.path(n,1)) + randn * 0.1;
                u = randn([1,10])*0.1;
                O.data.ultrasound(n) = O.path(n,2) - mean(O.terran(O.path(n,1)+u)) + randn * 0.1;
            end
            
            O.i = 1;
        end
        
        function flag = hasNext(O)
            if O.i >= size(O.path,1)
                flag = 0;
            else
                flag = 1;
            end
        end
        
        function [d, x] = read(O)
            d.infraredup = O.data.infraredup(O.i);
            d.infrareddown = O.data.infrareddown(O.i);
            d.ultrasound = O.data.ultrasound(O.i);
            x = O.path(O.i,1);
            O.i = O.i + 1;
        end
        
        function reset(O)
            O.i = 1;
        end
        
        function plot(O)
            left = min(O.path(:,1));
            right = max(O.path(:,1));
            x = left:0.01:right;
            hold on
            
            plot(x, O.roof(x))
            plot(x, O.terran(x))
            
            plot(O.path(:,1),O.path(:,2),'.')
        end
    end
end

