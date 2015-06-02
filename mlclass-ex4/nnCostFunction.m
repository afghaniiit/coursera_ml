function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1 - feedforward through the network
% ----------------------------------------------------------

% Add bias units to each of the input training sets
a1Bias = ones(size(X,1),1);
a1WithBias = [a1Bias , X];

% Multiply by theta to get z of next layer, use sigmoid to calculate
% activation
z2 = Theta1 * a1WithBias';
a2 = sigmoid(z2);
a2 = a2';

a2Bias = ones(size(a2, 1), 1);
a2WithBias = [a2Bias , a2];

% Repeat for last output layer
z3 = Theta2 * a2WithBias';
a3 = sigmoid(z3);

% This is our hypothesis (in vector for each training set)
h = a3;

% Need to convert y from [1;4;3;6;7] to [1000000000;0001000000 ...]
all_combos = eye(num_labels);
yMatrix = all_combos(y,:);    
    
J = (-(yMatrix.*log(h'))-(1-yMatrix).*log(1-h)')/m;
J = sum(J, 1);
J = sum(J, 2);

% -------------------------------------------------------------

% Add regularisation terms in too

theta1Squared = Theta1(:,2:end) .^ 2;
theta1SquaredSumRows = sum(theta1Squared, 1);
theta1SquaredSumCols = sum(theta1SquaredSumRows, 2);

theta2Squared = Theta2(:,2:end) .^ 2;
theta2SquaredSumRows = sum(theta2Squared, 1);
theta2SquaredSumCols = sum(theta2SquaredSumRows, 2);

theta1And2Sum = theta1SquaredSumCols + theta2SquaredSumCols;

J = J + ((lambda / (2 * m)) * (theta1And2Sum));


% -------------------------------------------------------------
% Back propagation steps here

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

Theta2NoBias = Theta2(:,2:end);

yMatrixT = yMatrix';

d3 = a3 - yMatrixT;
d2= Theta2NoBias' * d3;
d2 = d2 .* sigmoidGradient(z2);

D2 = D2 + (d3 * a2WithBias);
D1 = D1 + (d2 * a1WithBias);

% Added regularisation too .. 

Theta1_grad_reg = (lambda / m) * Theta1;
Theta2_grad_reg = (lambda / m) * Theta2;

Theta1_grad_reg(:,1) = zeros(hidden_layer_size,1);
Theta2_grad_reg(:,1) = zeros(num_labels,1);


Theta1_grad = (D1/m) + Theta1_grad_reg;
Theta2_grad = (D2/m) + Theta2_grad_reg;



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end