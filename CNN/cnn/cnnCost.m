function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolStride,lambda, pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolStride    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('theta','var')
    load cnnCost
    pred = false;
    poolStride = 2;
    lambda = 0.1;
end;

if ~exist('pred','var')    
    pred = false;
end;

imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolStride,numClasses);

Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

convDim = imageDim-filterDim+1; % dimension of convolved output
poolDim = floor(convDim / poolStride);
outputDim = (convDim) / poolStride; % dimension of subsampled output


activationsConvolved = cnnConvolve(filterDim, numFilters, images, Wc, bc);
activationsPooledMat = cnnPool(poolStride, activationsConvolved);
activationsPooled = reshape(activationsPooledMat,[],numImages);

h = bsxfun(@plus, Wd * activationsPooled, bd);
% h = (bsxfun(@minus, h, max(h,[], 1)));
%h = Wd * activationsPooled;

weightSum = sum(Wc(:) .^ 2) + sum(Wd(:) .^ 2);
probs = bsxfun(@rdivide,exp(h) , sum(exp(h),1));


% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(h,[],1);
    preds = preds';
    cost = 0;
    grad = 0;
    return;
end;

% groundTruth = full(sparse(labels, 1:numImages, 1));

groundTruth = zeros(size(probs));
for i=1:numImages
    groundTruth(labels(i),i) = 1;
end


weightDecay = lambda*(weightSum)/2;
cost = -sum(sum(groundTruth .* log(probs))) / numImages + weightDecay;


%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsamplings 
%  quickly.

dSoftmax = probs - groundTruth;
dPool = reshape(Wd' * dSoftmax, poolDim, poolDim, numFilters,numImages);

for i = 1 : numFilters
%     for j = 1 : numImages         
        pp = reshape(dPool(:,:,i,:),poolDim,poolDim,numImages);
        dActConv(:,:,i,:) = expand(pp, [poolStride poolStride 1]);        
%     end
end
% toc();
% logistic
% dActConv = (1 / (poolStride .^ 2)) * dActConv .* activationsConvolved .* (1 - activationsConvolved);
% tanh
dActConv = (1 / (poolStride .^ 2)) * dActConv .* (1 - (activationsConvolved .^ 2));


for filterNum = 1:numFilters            
%     temp = zeros(convolvedDim,convolvedDim); 
    
    for imageNum = 1:numImages    
%         im = images(:,:,imageNum);
        filter = reshape(dActConv(:,:,filterNum,imageNum),convDim,convDim);
        filter = rot90(squeeze(filter),2);
        Wc_grad(:,:,filterNum) = Wc_grad(:,:,filterNum) + conv2(images(:,:,imageNum), filter , 'valid');        
%         Wc_grad(:,:,filterNum) = conv2(im, filter , 'valid');        
    end
    
    Wc_grad(:,:,filterNum) = Wc_grad(:,:,filterNum) ./ numImages;
end

Wd_grad = dSoftmax * activationsPooled' ./ numImages + lambda .* Wd;
bd_grad = sum(dSoftmax,2) ./ numImages;

Wc_grad = Wc_grad + lambda * Wc;
dZConvTemp = sum(sum(mean(dActConv,4),1),2);
bc_grad = reshape(dZConvTemp, numFilters, 1);

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

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