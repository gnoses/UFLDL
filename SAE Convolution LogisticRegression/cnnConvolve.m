function convolvedFeatures = cnnConvolve(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)

if (exist('patchDim') == 0 )
    clc; clear all;
    load ('cnnConvolve.mat');
end

numImages = size(images, 4);
imageDimX = size(images, 1);
imageDimY = size(images, 2);
imageChannels = size(images, 3);
channelSize = size(W,2) / 3;
featureSize = sqrt(channelSize);
convolvedFeatures = zeros(numFeatures, numImages, imageDimX - patchDim + 1, imageDimY - patchDim + 1);

% Instructions:
%   Convolve every feature with every large image here to produce the 
%   numFeatures x numImages x (imageDim - patchDim + 1) x (imageDim - patchDim + 1) 
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(featureNum, imageNum, imageRow, imageCol) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + patchDim - 1, imageCol + patchDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 3 minutes 
%   Convolving with 5000 images should take around an hour
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)

% -------------------- YOUR CODE HERE --------------------
% Precompute the matrices that will be used during the convolution. Recall
% that you need to take into account the whitening and mean subtraction
% steps
WT = W * ZCAWhite;
precomputed = b - (WT * meanPatch);



% --------------------------------------------------------

convolvedFeatures = zeros(numFeatures, numImages, imageDimX - patchDim + 1, imageDimY - patchDim + 1);
for imageNum = 1:numImages % 8
  for featureNum = 1:numFeatures % 400

    % convolution of image with feature matrix for each channel
    %  57 * 57 convolved feature
    convolvedImage = zeros(imageDimX - patchDim + 1, imageDimY - patchDim + 1);
    for channel = 1:3

      % Obtain the feature (patchDim x patchDim) needed during the convolution
      % ---- YOUR CODE HERE ----
      % W [ 400 * 192 ]
%       feature = zeros(8,8); % You should replace this
      channelStart = (channelSize * (channel - 1)) + 1;
      feature = reshape(WT(featureNum,channelStart:channelStart + channelSize - 1),featureSize,featureSize); 
      
      
      
      % ------------------------

      % Flip the feature matrix because of the definition of convolution, as explained later
      feature = flipud(fliplr(squeeze(feature)));
      
      % Obtain the image
      % images [ row , col , 3 channel , n ]
      im = squeeze(images(:, :, channel, imageNum));

      % Convolve "feature" with "im", adding the result to convolvedImage
      % be sure to do a 'valid' convolution
      % ---- YOUR CODE HERE ----
      % W * T * x
      convolvedImage = convolvedImage + conv2(im, feature, 'valid');
      
      
      
      % ------------------------

    end
    
    % Subtract the bias unit (correcting for the mean subtraction as well)
    % Then, apply the sigmoid function to get the hidden activation
    % ---- YOUR CODE HERE ----
    convolvedImage = sigmoid(convolvedImage + precomputed(featureNum));
    
    
    
    % ------------------------
    
    % The convolved feature is the sum of the convolved values for all channels
    convolvedFeatures(featureNum, imageNum, :, :) = convolvedImage;
  end
end


end


function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
