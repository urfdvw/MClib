clear
close all
clc

rpt = report('test');
rpt.addtitle('test page')
rpt.addtext('this is a matrix')
rpt.addmatrix(ones(3))
rpt.addtext('this is a figure')
rpt.addimage('target2, method1.gif')
rpt.addtitle('The end')
rpt.close()