%%
% This script imports MaMuT tracking data into Imaris. 
% When running this script, ensure the folder ImarisCodeBase has been added
% to the folder path.

% In Imaris, the Scene icon must be selected, otherwise there may be
% problems with the connection between Matlab and Imaris. 

%%
% Use this section to correct for any flips or 90' rotations in X,Y, or Z.
% You may have to play around with individual transpositions or
% combinations before you find which one is which. 


transposeXY = true;
transposeXZ = false;
transposeYZ = false;
transposeinverse = false;

%%
% This section loads the MaMuT .xml into a .mat file that Imaris will read


trackingM = readMamutXML("D:\Caroline\140521_extended_gt_deduplicated.xml"); % Location of the MaMuT .xml file containing MaMuT tracks either from manual annotations or TGMM objects

if( transposeXY )
    trackingM(:,3:4) = trackingM(:,[4 3]);
end

if( transposeXZ )
    trackingM(:,3:5) = trackingM(:,[5 4 3]);
end

if( transposeYZ )
    trackingM(:,3:5) = trackingM(:,[3 5 4]);
end


if( transposeinverse )
    trackingM(:,3:5) = trackingM(:,[4 5 3]);
end

%%

stackRes = [1.0 1.0 5.0]; % X, Y, Z ratio for your dataset

%upload to IMaris
%update tracks to Imaris
addpath ImarisCodeBase

%create new spots entry in current Imaris scene
[vImarisApplication] = openImarisConnection;
aSurpassScene = vImarisApplication.GetSurpassScene;

aSpots = vImarisApplication.GetFactory.CreateSpots;
aSurpassScene.AddChild( aSpots, -1 );%add at the end
idx = aSurpassScene.GetNumberOfChildren;
%set current spot as selected
vImarisApplication.SetSurpassSelection(aSurpassScene.GetChild(idx-1));%0-indexing
%vImarisApplication.GetSurpassSelection.SetName(name); %uncomment this
%line if you want to define a name


parseCATMAIDdbToImarisSpots(trackingM, stackRes,[], []);
rmpath ImarisCodeBase
