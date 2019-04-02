clear
close all
clc
%% data
dt = 0.1;
vx = 1;
vy = 0;

x_c = 0;
y_c = 5;

i = 1;
while x_c <10
    path(i,:) = [x_c, y_c];
    x_c = x_c + vx*dt;
    y_c = y_c + vy*dt;
    i = i + 1;
end
%% read sensor
s = sensors(@roof,@terran,path);
figure
s.plot()
figure
while s.hasNext()
    [d, x_c] = s.read();
    subplot(3,1,1)
    hold on
    plot(x_c,d.infraredup,'.')
    subplot(3,1,2)
    hold on
    plot(x_c,d.infrareddown,'.')
    subplot(3,1,3)
    hold on
    plot(x_c,d.ultrasound,'.')
end

%% modle function
function y = roof(x)
y = 10*ones(size(x));
y(x>5 & x<7) = 9;
end
function y = terran(x)
y = zeros(size(x));
y(x>2 & x<3) = 2;
end