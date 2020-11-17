%basic example on how to connect to running Imaris session and manipulate
%data
%function basicImarisCommunication()

%---------------------------------------------------------
%-----basic interface to setup the pipe betwen Matlab and Imaris----------
% connect to Imaris interface: Imaris should be running!!!
javaaddpath C:\Program' Files'\Bitplane\Imaris' x64 9.5.0'\XT\Matlab\ImarisLib.jar
vImarisLib = ImarisLib;
vServer = vImarisLib.GetServer;

if(vServer.GetNumberOfObjects~=1)
    error 'Either Imaris is not running or you are running more than one instance. Not allowed in this context.'
end
vObjectId=vServer.GetObjectID(0);
vImarisApplication = vImarisLib.GetApplication(vObjectId);
 
%-------------------------------------------------------------------

%now I can run any command
aSurface = vImarisApplication.GetFactory.ToSurfaces(vImarisApplication.GetSurpassSelection);
 aSurfaceIndex = 0;
 aTimeIndex = aSurface.GetTimeIndex(aSurfaceIndex)
 
 