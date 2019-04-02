clear
close all
clc
addpath(genpath('C:\Users\hechu\Documents\GitHub\MClib'))%% data
%%
dt = 0.1;
vx = 1;
vy = 0;

x_c = 0;
y_c = 7;

i = 1;
while x_c <10
    path(i,:) = [x_c, y_c];
    x_c = x_c + vx*dt;
    y_c = y_c + vy*dt;
    i = i + 1;
end
%% doing smoothing while reading sensor
s = sensors(@roof,@terran,path);
filter = FilterAverage();
filter.setThreshold(0.3);
figure
s.plot()
hold on
while s.hasNext()
    [d, x_c] = s.read();
    height = filter.estimate(0.5*(d.infrareddown + d.ultrasound));
    plot(x_c, height,'+')
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
%% filter modle
function p = LiOb(y,x)
sig = 0.5;
y(1)
end
function RndTr
end
