classdef report < handle
    % Report generator class
    
    properties
        rpt % report object
    end
    
    methods
        function O = report(filename)
            import mlreportgen.report.*
            O.rpt = Report(filename,'html-file');
        end
        
        function O = addmatrix(O,A)
            if numel(A) == 1
                O.addtext(num2str(A))
            else
                import mlreportgen.dom.*
                tableStyle = { ...
                    Border('solid','black','1px'), ...
                    ColSep('solid','black','1px'), ...
                    RowSep('solid','black','1px') ...
                    };
                table = Table(A);
                table.TableEntriesHAlign = 'center';
                table.Style = tableStyle;
                add(O.rpt,table)
            end
        end
        
        function O = addimage(O,imagefile)
            import mlreportgen.report.*
            add(O.rpt,FormalImage('Image',which(imagefile)));
        end
        
        function O = addtext(O,text)
            import mlreportgen.report.*
            add(O.rpt,text)
        end
        
        function O = addtitle(O,text)
            import mlreportgen.report.*
            C = Chapter(text);
            add(O.rpt,C)
        end
        
        function close(O)
            close(O.rpt);
            rptview(O.rpt);
        end
    end
end

