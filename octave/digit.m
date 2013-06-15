%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

% Initialization
%clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 784;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
hidden_layer_2_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

%load('train.csv'); % training data stored in arrays X, y
X = train(:,2:end);
y = train(:,1);
m = size(X, 1);
Y = zeros(m, num_labels);
for i=1:num_labels
  Y(:, i)= y==i-1;
end

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:144), :);

displayData(sel);


% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_2_size);
initial_Theta3 = randInitializeWeights(hidden_layer_2_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:)];
%initial_nn_params = [Theta1(:) ; Theta2(:) ; Theta3(:)];

fprintf('Initialized Theta\n');
fflush(stdout);

options = optimset('MaxIter', 100);

% Set regularization parameter lambda to 1 (you should vary this)

lambda = 0.3;

costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, hidden_layer_2_size, num_labels, X, Y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
fprintf('Finished Iterations\n');
fflush(stdout);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))): ...
								hidden_layer_size * (input_layer_size + 1) + hidden_layer_2_size * (hidden_layer_size + 1)),  ...
								hidden_layer_2_size,  ...
								(hidden_layer_size + 1));

Theta3 = reshape(nn_params((1 + hidden_layer_size * (input_layer_size + 1) + hidden_layer_2_size * (hidden_layer_size + 1)):end), ...
                 num_labels, (hidden_layer_2_size + 1));

fprintf('Predicting\n');
fflush(stdout);

pred = predict(Theta1, Theta2, Theta3, X)-1;

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%load('test.csv');
result = predict(Theta1, Theta2, Theta3, test) -1;

fid=fopen('result3.csv', 'w');
fprintf(fid,'%d\n',result);
fclose(fid);

rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, Theta3, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end

