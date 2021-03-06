function [cost] = nnTest(theta, visibleSize, hiddenSize, outputSize, data, output,lambda)



W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
index = hiddenSize*visibleSize+outputSize*hiddenSize;
W2 = reshape(theta(hiddenSize*visibleSize+1:index), outputSize, hiddenSize);

b1 = theta(index+1:index+hiddenSize);
b2 = theta(index+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

elementSize = size(data,2); 

groundTruth = full(sparse(output, 1:elementSize, 1));


% forward propagation
a2 = sigmoid(bsxfun(@plus, W1 * data(:,1:elementSize), b1));
a3 = sigmoid(bsxfun(@plus, W2 * a2, b2));

error = sum(sqrt(sum((a3 - groundTruth) .* (a3 - groundTruth))));
CostTerm = error / elementSize / 2;



%weightDecay = lambda*(sum(W1(:).*W1(:)) + sum(W2(:).*W2(:)))/2;
cost = CostTerm;% + weightDecay;

% find true positive
[maxValue, selection] = max(a3, [], 1);

eval = sum(selection' == output);

fprintf('%f correct %d/%d (%.2f %%)\n', cost, eval, elementSize, eval / elementSize * 100);




wrong = selection' ~= output;
wrongi = find(wrong);
wrongiPartial = wrongi(1:25);
% displayColorNetwork(data(:,wrongiPartial),sqrt(784));
% disp([selection(wrongiPartial)', output(wrongiPartial)]);
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

