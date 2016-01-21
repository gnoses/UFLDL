function convolvedFeatures = cnnConvolve(filterDim, numFilters, images, W, b)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

numImages = size(images, 3);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);

% Instructions:
%   Convolve every filter with every image here to produce the 
%   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 30 seconds 
%   Convolving with 5000 images should take around 2 minutes
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)


for imageNum = 1:numImages
  for filterNum = 1:numFilters

    filter = W(:,:,filterNum);
    % Flip the feature matrix because of the definition of convolution, as explained later
    filter = rot90(squeeze(filter),2);
      
    % Obtain the image
    im = squeeze(images(:, :, imageNum));
    convolvedImage = ActFunc(conv2(im, filter, 'valid') + b(filterNum), 'tanh');
   
    convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
  end
end


end


function out = ActFunc(x, type)
    
    % sigmoid
    if strcmp(type, 'logistic')
        out = 1 ./ (1 + exp(-x));
    elseif strcmp(type, 'tanh')
        out = tanh(x);
    elseif strcmp(type, 'relu')
        out = log(1+exp(x));
    end
    % tanh
%     out = tanh(x)
end


function out = DerivActFunc(x, type)
  
    % sigmoid
    if strcmp(type, 'logistic')
        out = ActFunc(x,type) .* (1 - ActFunc(x,type));
    elseif strcmp(type, 'tanh')
        out = 1 - (tanh(x) .^ 2);
    elseif strcmp(type, 'relu')
        out = 1 ./ (1 + exp(-x));
    end
    
    
    % tanh
%     out = tanh(x)
end