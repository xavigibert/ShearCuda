function GPUtestLOG(text, iserror)
% GPUtestLOG writes to the log file
% GPUtestLOG writes to the log file, or generates an error if the flag
% iserror = 1

%
%     Copyright (C) 2012  GP-you Group (http://gp-you.org)
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

% if printdisp = 1 writes also to standard output
printdisp = GPUtest.printDisplay;

if GPUtest.stopOnError==1
  if (iserror)
    error(text);
  end
end

fid = fopen(GPUtest.logFile,'a+');

[ST ,I] = dbstack(1);
if (iserror)
  for i=1:length(ST)
    str = sprintf('*** Error in file %s, line %d\n', ST(i).file, num2str(ST(i).line));
    fprintf(fid,str);
    if printdisp
      disp(str);
    end
  end
  str = sprintf('*** Error %s\n',text);
  fprintf(fid,str);
  if printdisp
    disp(str);
  end
else
  fprintf(fid,'%s\n',text);
  if printdisp
    disp(text);
  end
end
fclose(fid);


end


