function GPUmatLoadModules(path)

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

%

% save current dir
curdir = pwd;

% finds all subdirectories
subs = getsubdirs(path);

% look for moduleinit.m file
for i=1:length(subs)
  
  file = fullfile(subs{i},'moduleinit.m');
  if (exist(file,'file'))
    %disp(['Loading module in -> ' subs{i}]);
    try
      cd(subs{i});
      moduleinit;
      addpath(pwd);
      cd(curdir);
    catch
      cd(curdir);
      error(lasterror);
    end
  end
  
  
end














function r = getsubdirs(path)

p = genpath(path);

% have to split using pathsep
remain = p;

r = {};
while ~isempty(remain)
  [token, remain] = strtok(remain,pathsep);
  if (~isempty(token))
    r{end+1} = token;
  end
end

