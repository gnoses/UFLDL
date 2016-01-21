function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);
numPooledFeatures = floor(convolvedDim / poolDim);
pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.
filter = ones(convolvedDim);

% for imageNum = 1:numImages
%   for filterNum = 1:numFilters
%     pooledFeatures = conv2(convolvedFeatures(:,:,filterNum,imageNum), filter, 'invalid');
%     
%   end
% end

for row=0:numPooledFeatures-1
    startRow = row*poolDim+1;
    endRow = startRow +  poolDim - 1;
    for col=0:numPooledFeatures-1
        startCol = col*poolDim+1;
        endCol = startCol +  poolDim - 1;
        pooledFeaturesSingle = convolvedFeatures(startRow:endRow,startCol:endCol,:,:);
        
        % mean pooling
        pooledFeatures(row+1,col+1,:,:) = mean(mean(pooledFeaturesSingle,1),2);
        
        % max pooling
%         pooledFeatures(:,:,row+1,col+1) = max(max(pooledFeaturesSingle,[], 4),[],3);
    end
end

end

