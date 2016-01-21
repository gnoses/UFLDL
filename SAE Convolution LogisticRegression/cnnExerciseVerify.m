

imageChannels = 3;     % number of channels (rgb, so 3)

patchDim = 8;          % patch dimension

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize = visibleSize;   % number of output units
hiddenSize = 400;           % number of hidden units 

epsilon = 0.1;	       % epsilon for ZCA whitening

poolDim = 2;          % dimension of pooling region


% displayColorNetwork( (W*ZCAWhite)');
load('STL10Features.mat', 'optTheta', 'ZCAWhite', 'meanPatch');
W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

% For 1000 random points


cat = imread('catSample.bmp');
convImages = double(cat ./ 255);
%% Use only the first 8 images for testing
% load stlTrainSubset.mat % loads numTrainImages, trainImages, trainLabels
% convImages = trainImages(:, :, :, 1:8); 

imageDimX = size(convImages, 1);
imageDimY = size(convImages, 2);
numImages = size(convImages, 4);

convolvedFeatures = cnnConvolve(patchDim, hiddenSize, convImages, W, b, ZCAWhite, meanPatch);

% test random 1000 locations
for i = 1:1000
    featureNum = randi([1, hiddenSize]);
    imageNum =  randi([1, numImages]);
    imageRow = randi([1, imageDimX - patchDim + 1]);
    imageCol = randi([1, imageDimY - patchDim + 1]);    

    patch = convImages(imageRow:imageRow + patchDim - 1, imageCol:imageCol + patchDim - 1, :, imageNum);
    patch = patch(:);            
    patch = patch - meanPatch;
    patch = ZCAWhite * patch;

    features = feedForwardAutoencoder(optTheta, hiddenSize, visibleSize, patch); 

    if abs(features(featureNum, 1) - convolvedFeatures(featureNum, imageNum, imageRow, imageCol)) > 1e-9
        fprintf('Convolved feature does not match activation from autoencoder\n');

    fprintf('Feature Number    : %d\n', featureNum);0
    fprintf('Image Number      : %d\n', imageNum);
    fprintf('Image Row         : %d\n', imageRow);
    fprintf('Image Column      : %d\n', imageCol);
    fprintf('Convolved feature : %0.5f\n', convolvedFeatures(featureNum, imageNum, imageRow, imageCol));
    fprintf('Sparse AE feature : %0.5f\n', features(featureNum, 1));       
    error('Convolved feature does not match activation from autoencoder');
    end 
end
disp('Congratulations! Your convolution code passed the test.');

pooledFeatures = cnnPool(poolDim, convolvedFeatures);
testMatrix = reshape(1:64, 8, 8);
expectedMatrix = [mean(mean(testMatrix(1:4, 1:4))) mean(mean(testMatrix(1:4, 5:8))); ...
                  mean(mean(testMatrix(5:8, 1:4))) mean(mean(testMatrix(5:8, 5:8))); ];

testMatrix = reshape(testMatrix, 1, 1, 8, 8);

pooledFeatures = squeeze(cnnPool(4, testMatrix));

if ~isequal(pooledFeatures, expectedMatrix)
    disp('Pooling incorrect');
    disp('Expected');
    disp(expectedMatrix);
    disp('Got');
    disp(pooledFeatures);
else
    disp('Congratulations! Your pooling code passed the test.');
end


