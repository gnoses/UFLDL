function [cost,grad] = nnCost(theta, visibleSize, hiddenSize, outputSize, data, output,lambda)


if (exist('theta') == 0 )
    clc; clear all;
    load ('nnCost.mat');
end

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
index = hiddenSize*visibleSize+outputSize*hiddenSize;
W2 = reshape(theta(hiddenSize*visibleSize+1:index), outputSize, hiddenSize);

b1 = theta(index+1:index+hiddenSize);
b2 = theta(index+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

elementSize = size(data,2); 
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
groundTruth = full(sparse(output, 1:elementSize, 1));


% forward propagation
a2 = sigmoid(bsxfun(@plus, W1 * data(:,1:elementSize), repmat(b1, 1, elementSize)));
a3 = sigmoid(bsxfun(@plus, W2 * a2, repmat(b2, 1,elementSize)));

% back propagation
d3 = (a3 - groundTruth) .* a3 .* (1-a3);
d2 = (W2' * d3) .* a2 .* (1-a2);

W2grad = (d3 * a2') / elementSize;
W1grad = (d2 * data') / elementSize;
b1grad = sum(d2,2) / elementSize;
b2grad = sum(d3,2) / elementSize;

error = sum(sqrt(sum((a3 - groundTruth) .* (a3 - groundTruth))));
CostTerm = error / elementSize / 2;



weightDecay = lambda*(sum(W1(:).*W1(:)) + sum(W2(:).*W2(:)))/2;
cost = CostTerm;% + weightDecay;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

%fprintf('%f = %f + %f\n', cost,CostTerm, weightDecay);
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

