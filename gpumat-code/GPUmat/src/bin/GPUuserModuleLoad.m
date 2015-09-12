% GPUuserModuleLoad - Loads CUDA .cubin module
% 
% SYNTAX
% 
% GPUuserModuleLoad(module_name, filename)
% module_name - string
% filename - string
% 
% 
% MODULE NAME
% na
% 
% DESCRIPTION
% GPUuserModuleLoad(module name, filename) loads the CUDA
% .cubin module (filename) and assigns to it the name module name.
% Module handler can be retrieved using GPUgetUserModule.
% 
% EXAMPLE
% 
% %GPUuserModuleLoad('numerics','.\numerics.cubin')
