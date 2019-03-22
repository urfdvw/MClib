clear
close all
clc


t = 0:0.01:pi;
gifm = gifmaker('test.gif');
for i = 1:10
    plot(t,sin(t*(1+0.1*i)))
    pause(0.1)
    gifm.capture()
end