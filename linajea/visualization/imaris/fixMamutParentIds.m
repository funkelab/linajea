function trackingMatrix = fixMamutParentIds(trackingMatrix)

parentIdsUnfixed = trackingMatrix(1:end, 7);

for i = 1:length(trackingMatrix)
    
    currentParentId = parentIdsUnfixed(i);
    
    for j = 1:length(trackingMatrix)
        
        if trackingMatrix(j,1) == currentParentId && j > i
            
            trackingMatrix(j,7) = trackingMatrix(i,1);
            
        elseif trackingMatrix(j,8) == 0
                
            trackingMatrix(j,7) = -1;
        
        end
        
    end
end
            
            
    