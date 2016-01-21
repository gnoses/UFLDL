function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
if (exist('softmaxModel', 'var') == 0)
    load('softmaxPredict');
end

trainingDataSize = size(data,2);
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
theta = reshape(theta, softmaxModel.numClasses, softmaxModel.inputSize);
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

for i=1:trainingDataSize    
    prob = exp(theta * [1;data(:,i)]);
    prob = prob / sum(prob);
    [maxValue, pred(i)] = max(prob);
end


% ---------------------------------------------------------------------

end

