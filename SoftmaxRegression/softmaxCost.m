function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%


% self initialization
if (exist('theta', 'var') == 0)    
    clear all; clc; close all;
    load ('softMaxCost');
    lambda = 1e-4;
end 

trainingDataSize = size(data,2);
theta = reshape(theta, numClasses, inputSize);

%groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


for i=1:trainingDataSize
    singleData = data(:,i);
    j = labels(i);    
    prob = exp(theta * singleData);
    prob = prob / sum(prob);
    cost = cost + log(prob(j));
    
    select = zeros(numClasses,1);
    select(j) = 1;

    thetagrad = thetagrad + ((select - prob) * singleData');
    
    %weightDecay = weightDecay + sum(theta .* theta);
end

weightDecay = sum(sum(theta .* theta));
cost = -cost / trainingDataSize + lambda/2*weightDecay;
thetagrad = - thetagrad / trainingDataSize + lambda * theta;


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

