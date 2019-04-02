clear
close all
clc
addpath(genpath('..\..'))

rpt = report('PMCs');

for pdf_case = 1:2
    for meth_case = 1:6
        PMC_test
        rpt.addtitle(gifname)
        rpt.addimage(gifname)
        rpt.addtext(pdfnames{pdf_case})
        rpt.addtext(pmc.info())
        keep rpt pdf_case meth_case
    end
end
rpt.close()