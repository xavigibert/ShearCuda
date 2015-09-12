function y=locateCL
path = getenv('PATH');


pathCell = splitPath(path);
CL_INCLUDE = '';

for i=1:length(pathCell)
  pi = pathCell{i};
  % check for CUDA dlls
  if (exist(fullfile(pi,'cl.exe'),'file'))
    if (exist(fullfile(pi,'..','include'),'dir'))
      CL_INCLUDE = [' -I"' fullfile(pi,'..','include') '" '];
    end
  end
  
end

y = CL_INCLUDE;

end

function y = splitPath(path)
if (ispc)
  c = ';';
end
if (isunix)
  c = ':';
end

y={};
[t,r] = strtok(path,c);
y{end+1} = t;
while ~isempty(r)
  [t,r] = strtok(r,c);
  y{end+1} = t;
end


end