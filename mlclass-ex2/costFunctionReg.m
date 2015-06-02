function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% COST FUNCTION

% Calculate the hypothesis first (using sigmoid)
hLinear = theta' * X';
hLogistic = sigmoid(hLinear);
hLogistic = hLogistic'; % Convert into column vector for below

% Calculate the sum, then sum over i
sumTerm = (-y .* log(hLogistic)) - ((1 - y) .* log(1 - hLogistic));

J = (1/m) * sum(sumTerm, 1) + ((lambda / (2 *m)) * sum(theta(2:end)'.^ 2, 2));


% PARTIAL DERIVATIVES

% pdSumTerm(1) = (1/m) * sum(  (hLogistic - y)  .* X(:,1), 1);
% pdSumTerm(2) = (1/m) * sum(  (hLogistic - y)  .* X(:,1), 2);
% pdSumTerm(3) = (1/m) * sum(  (hLogistic - y)  .* X(:,1), 3);


grad = ((1/m) * ((hLogistic - y)'  * X))' + ((lambda / m) * ([0, theta(2:end)']'));

% Add extra regularisation term here 




% =============================================================

end
