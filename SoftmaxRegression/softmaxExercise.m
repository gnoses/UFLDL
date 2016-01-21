%% CS294A/CS294W Softmax Exercise 
clc ; clear all; close all;

inputSize = 28 * 28 + 1; % Size of input vector (MNIST images are 28x28)
numClasses = 10;     % Number of classes (MNIST images fall into 10 classes)

lambda = 1e-4; % Weight decay parameter


inputData = loadMNISTImages('mnist/train-images-idx3-ubyte');
labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

%load('inputData2');
%load('labels2');


trainingDataSize = size(inputData,2);
bias = ones(1,trainingDataSize);
inputData = inputData(:,1:trainingDataSize);
labels = labels(1:trainingDataSize);
inputData = [bias; inputData];

%image = reshape(inputData(:,4), sqrt(inputSize), sqrt(inputSize));
%colormap('gray');
%imagesc(image);

labels(labels==0) = 10; % Remap 0 to 10

DEBUG = false; % Set DEBUG to true when debugging.




if (0)
    for i=1:200

        % Randomly initialise theta
        theta = 0.005 * randn(numClasses * inputSize, 1);
        
        [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
        theta = theta - grad;
        
        disp([i , cost]);
%         plot(theta, '-b');
%         drawnow;
    end
else
    %load ('theta');
%     softmaxModel.optTheta = theta;
%     softmaxModel.inputSize = inputSize;
%     softmaxModel.numClasses = numClasses;
end 



    

if DEBUG
    theta = 0.005 * randn(numClasses * inputSize, 1);
        
    [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);
    %theta = theta - grad;
        
    numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
                                    inputSize, lambda, inputData, labels), theta);
%     numgrad = computeNumericalGradient(theta, visibleSize, ...
%         hiddenSize, lambda, ...
%         sparsityParam, beta, ...
%         patches,grad);
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]); 
    
    % Compare numerically computed gradients with those computed analytically
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
    % The difference should be small. 
    % In our implementation, these values are usually less than 1e-7.

    % When your gradients are correct, congratulations!
end

%%======================================================================
%% STEP 4: Learning parameters
%
%  Once you have verified that your gradients are correct, 
%  you can start training your softmax regression code using softmaxTrain
%  (which uses minFunc).

options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, labels, options);
%                           
% Although we only use 100 iterations here to train a classifier for the 
% MNIST data set, in practice, training for more iterations is usually
% beneficial.

%%======================================================================
%% STEP 5: Testing
%
%  You should now test your model against the test images.
%  To do this, you will first need to write softmaxPredict
%  (in softmaxPredict.m), which should return predictions
%  given a softmax model and the input data.

images = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
labels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

inputData = images;

% You will have to implement softmaxPredict in softmaxPredict.m
[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% After 100 iterations, the results for our implementation were:
%
% Accuracy: 92.200%
%
% If your values are too low (accuracy less than 0.91), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
