classdef gifmaker < handle
    % Gif generator class
    properties
        h
        filename
        i
    end
    
    methods
        function O = gifmaker(filename)
            % initialize function, used insted of figure to create a new
            % figure to capture
            O.h = figure;
            axis tight manual % this ensures that getframe() returns a consistent size
            O.filename = filename;
            O.i = 1;
        end
        
        function capture(O)
            drawnow % force draw, incase no pause was used
            % Capture the plot as an image
            frame = getframe(O.h);
            im = frame2im(frame);
            [imind,cm] = rgb2ind(im,256);
            % Write to the GIF File
            if O.i == 1
                imwrite(imind,cm,O.filename,'gif', 'Loopcount',inf);
            else
                imwrite(imind,cm,O.filename,'gif','WriteMode','append');
            end
            O.i = O.i + 1;
        end
    end
end

