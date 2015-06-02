function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Calculate the J(theta) cost function first
h = theta' * X';
h = h';

leftSum =  (1/ (2*m)) * sum(((h - y).^2), 1);
rightSum = (lambda / (2*m)) * sum((theta(2:end).^2), 1);

J = leftSum + rightSum;

% Now calculate the partial derivatives

% Calculate the general case parital derivative with regularisation
% Then overwrite the Theta0 (Theta(1)) with non regularised version

dJSumTerm = ((h - y)' * X);

dJ = ( (1 / m) * dJSumTerm)' + ( (lambda/m) * theta);
dJ(1) = ( (1 / m) * dJSumTerm(1));

grad = dJ;






% =========================================================================

grad = grad(:);

end
