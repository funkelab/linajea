function vImarisApplication=openImarisConnection()
%---------------------------------------------------------
%-----basic interface to setup the pipe betwen Matlab and Imaris----------
% connect to Imaris interface: Imaris should be running!!!
p = 'C:\Program Files\Bitplane\Imaris x64 9.5.0\XT\Matlab\ImarisLib.jar';
if ~ismember(p, javaclasspath)
    javaaddpath(p) 
end
vImarisLib = ImarisLib;
vServer = vImarisLib.GetServer;

objN=0;

 if(vServer.GetNumberOfObjects>2)
     error 'Either Imaris is not running or you are running more than one instance. Not allowed in this context.'
 elseif(vServer.GetNumberOfObjects==2)
     disp 'WARNING: using second Imaris object (most likely another user has also opened Imaris'
     objN=1;
 end


vObjectId=vServer.GetObjectID(objN);
vImarisApplication = vImarisLib.GetApplication(vObjectId);
 