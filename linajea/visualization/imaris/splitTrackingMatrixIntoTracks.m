function tracks = splitTrackingMatrixIntoTracks(trackingMatrix)

trackIds = zeros(1,1);

for i = 1:length(trackingMatrix)
    if trackingMatrix(i,8) == 0
        trackIds(i) = trackingMatrix(i,10);
    end
end

trackNumber = length(trackIds);

tracks = cell(trackNumber,1);

for i = 1:length(trackIds)
    n = 1;
    currentTrackId = trackIds(i);
    for j = 1:length(trackingMatrix)
        if trackingMatrix(j,10) == currentTrackId
            tracks{i}(n,1:10) = trackingMatrix(j,1:10);
            n = n + 1;
        end
    end
end
