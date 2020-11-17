%changes porperties of all spots in Imaris in a surpass scene (useful if we
%use one spot per lineage)

%INPUT:
%color:  [R,G,B,alpha] all between [0,1] (alpha = 0 most of teh time)
function setSpotsAllProperties(property,color,radius)


[vImarisApplication]=openImarisConnection;
aSurpassScene = vImarisApplication.GetSurpassScene;
numCh = aSurpassScene.GetNumberOfChildren;

for kk = 0 : numCh-1
   aSpots = aSurpassScene.GetChild( kk );
   
   %check if it is a spot
   if( vImarisApplication.GetFactory.IsSpots( aSpots ) == false )
       continue;
   end
   
   switch(property)
       case 'track' %change thickness and color from the tracks
           
           
       case 'points'%change thickness and color from the points
           vRGBA = round(color * 255); % need integer values scaled to range 0-255
           vRGBA = uint32(vRGBA * [1; 256; 256*256; 256*256*256]); % combine different components (four bytes) into one integer
           aSpots.SetColorRGBA(vRGBA);
           vSpots = vImarisApplication.GetFactory.ToSpots( aSpots );
           aux = vSpots.GetRadiiXYZ;
           aux = radius * ones(size(aux));
           vSpots.SetRadiiXYZ( aux );
       case 'uncheck'
           aSpots.SetVisible( false );
       case 'check'
           aSpots.SetVisible( true );
       otherwise
           disp(['ERROR: Property ' property ' not define in the code']);
   end
   
   break;
end