function runme

%% test all .m files
ls=dir('ex*.m');
for i=1:length(ls)
  n = ls(i).name(1:end-2);
  eval(n);
end


end

