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
   

theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
theta = reshape(theta, softmaxModel.numClasses, softmaxModel.inputSize);

probAll = exp(theta * data);
prob = probAll ./ repmat(sum(probAll),4,1);
[maxValue, pred] = max(prob, [], 1);
% [maxValue, pred] = max(probAll, [], 1);
% ---------------------------------------------------------------------

end
