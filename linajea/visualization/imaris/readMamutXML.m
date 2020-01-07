%we return the same structure as database from CATMAID so we can recycle
%code

%OUTPUT: trackingMatrix Nx10 array, where N is the number of spots.
% We follow the SWC convention (more or less, since we need time): [id, type, x, y, z, radius, parent_id, time, confidence, skeletonId]
function trackingMatrix = readMamutXML(filenameXML)


%read xml file
xDoc = xmlread(filenameXML);


%extract spot information
NodeList = xDoc.getElementsByTagName('Spot');
N = NodeList.getLength();

trackingMatrix = zeros(N,10);

for ii=1:N 
    fstNode = NodeList.item(ii-1);%contains the ii=th Gaussian Mixture
    
    attrs = fstNode.getAttributes();
    
    trackingMatrix(ii,:) = [getNumericalAttribute(attrs,'ID'), -1, getNumericalAttribute(attrs,'POSITION_X'), getNumericalAttribute(attrs,'POSITION_Y'),...
                      getNumericalAttribute(attrs,'POSITION_Z'), getNumericalAttribute(attrs,'RADIUS'), -1, getNumericalAttribute(attrs,'FRAME'),...
                      getNumericalAttribute(attrs,'QUALITY'), -1];  
end

%extract linake information (parent_id and skeleton_id)
treenodeIdMap = containers.Map(trackingMatrix(:,1),[1:N]);

NodeList = xDoc.getElementsByTagName('Track');
Tr = NodeList.getLength();

for ii = 1:Tr
    
    fstTr = NodeList.item(ii-1);%contains the ii=th Gaussian Mixture
    skeletonId = getNumericalAttribute(fstTr.getAttributes(), 'TRACK_ID' );
    
    %parse all the edges inside a track
    fstList = fstTr.getElementsByTagName('Edge');
    
    E = fstList.getLength();
    for jj = 1:E
        fstEdge = fstList.item(jj-1);
       attr =  fstEdge.getAttributes();
       parId = getNumericalAttribute(attr, 'SPOT_SOURCE_ID');
       chId = getNumericalAttribute(attr, 'SPOT_TARGET_ID');
       
       trackingMatrix(treenodeIdMap(parId),10 ) = skeletonId;
       trackingMatrix(treenodeIdMap(chId),10 ) = skeletonId;
       trackingMatrix(treenodeIdMap(chId),7 ) = parId;
       
%        %self-verification: parent shuld always be earlier time point
%        if( trackingMatrix(treenodeIdMap(parId),8) >= trackingMatrix(treenodeIdMap(chId),8) )
%            warning 'Parent has a later time point than child. It should not happen'
%        end
    end
end

%sort elements in time
trackingMatrix = sortrows(trackingMatrix,8);
