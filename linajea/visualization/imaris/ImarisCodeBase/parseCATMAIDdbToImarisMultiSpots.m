
% INPUT:
% 
%
%trackingM:               array Nx 10 retrived from tracking information in CATMAID databse
%information from CATMAID database
%stackRes:                array 3x1 indicating resolution (needed to parse xyz from CATMAID (in um) to pixels    
%sourceVolumeDimensions:  array 3x1 indicating dimensions of image stack.
%Use [] if you did not scale the volume in Imaris
%targetVolumeDimensions:  array 3x1 to use "fake" imaris volume. Use [] if
%the volume at Imaris has the same dimensions as sourceVolumeDimensions


%trackingM matrix contains teh following columns
%id, type, x, y, z, radius, parent_id, time, confidence, skeleton_id
function [skeletonIdVec, spotIdxVec] = parseCATMAIDdbToImarisMultiSpots(trackingM, stackRes,sourceVolumeDimensions, targetVolumeDimensions)

skeletonIdVec = unique( trackingM(:,10 ) );

spotIdxVec = zeros(length(skeletonIdVec),1);
for kk = 1: length(skeletonIdVec)
    skeletonId = skeletonIdVec(kk);
    trackingMaux = trackingM( trackingM(:,10) == skeletonId,: );
    
    %create new spots entry in current Imaris scene
    
    [vImarisApplication]=openImarisConnection;
    aSurpassScene = vImarisApplication.GetSurpassScene;
    
    aSpots = vImarisApplication.GetFactory.CreateSpots;
    aSurpassScene.AddChild( aSpots, -1 );%add at the end
    idx = aSurpassScene.GetNumberOfChildren;
    spotIdxVec(kk) = idx;
    %set current spot as selected
    vImarisApplication.SetSurpassSelection(aSurpassScene.GetChild(idx-1));%0-indexing
    %vImarisApplication.GetSurpassSelection.SetName(name); %uncomment this
    %line if you want to define a name
    
    %update spots
    parseCATMAIDdbToImarisSpots(trackingMaux, stackRes, sourceVolumeDimensions, targetVolumeDimensions);
    
    %disp(['Spot ' num2str(kk) ' to skeletonId ' num2str(skeletonId)]);
end
