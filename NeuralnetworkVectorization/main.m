
clc ; clear all; close all;

%inputData = loadMNISTImages('mnist/train-images-idx3-ubyte');
%labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

load('inputData2');
load('labels2');
image = reshape(inputData(:,4), sqrt(784), sqrt(784));

%colormap('gray');
%imagesc(image);

visibleSize = size(inputData, 1);
hiddenSize = 50;
sparsityParam = 0.1;
outputSize = 10;
lambda = 3e-3; % Weight decay parameter

labels(labels==0) = 10; % Remap 0 to 10
m = size(inputData, 2);

if (1)
    load('trainedNet');
    [cost, grad] = nnCost(opttheta, visibleSize, hiddenSize, outputSize, inputData,labels, lambda);
    exit;
end

theta = initializeParameters(hiddenSize, visibleSize, outputSize);

% for i=1:m
%     a2 = W1 * inputData
% end
if (0)
    [cost, grad] = nnCost(theta, visibleSize, hiddenSize, outputSize, inputData,labels, lambda);

    numgrad = computeNumericalGradient( @(p) nnCost(p, visibleSize, hiddenSize, outputSize, inputData,labels,lambda), ...                                   
                              theta);

    % Use this to visually compare the gradients side by side
    %numgrad = numgrad ./ norm(numgrad,2);
    %grad = grad ./ norm(grad,2);
    disp([numgrad grad]); 
   
    % Compare numerically computed gradients with the ones obtained from backpropagation
    diff = norm(numgrad-grad)/norm(numgrad+grad);
    disp(diff); % Should be small. In our implementation, these values are
                % usually less than 1e-9.

                % When you got this working, Congratulations!!! 
    pause;
end

% Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 200;	  % Maximum number of iterations of L-BFGS to run 
minFuncOptions.display = 'on';

[opttheta, cost] = minFunc( @(p) nnCost(p, visibleSize, hiddenSize, outputSize, inputData,labels,lambda), ...                                   
                              theta, options);
save('trainedNet', 'opttheta');
[cost, grad] = nnCost(opttheta, visibleSize, hiddenSize, outputSize, inputData,labels, lambda);

cost                          
%plot (inputData(:,10), 'r-');
%hold on;
%plot (a3(:,10), 'b-');

%inputData2 = inputData(:,1:10000)