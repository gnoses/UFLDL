
clc ; clear all; close all;

%inputData = loadMNISTImages('mnist/train-images-idx3-ubyte');
%labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

load('inputData2');
load('labels2');
image = reshape(inputData(:,4), sqrt(784), sqrt(784))

colormap('gray');
imagesc(image);

%inputData2 = inputData(:,1:10000)