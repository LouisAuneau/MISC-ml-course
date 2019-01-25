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

J = 1/(2*m) * sum(((X * theta) - y) .^ 2); 
J = J + ((lambda / (2 * m)) * sum((theta .^ 2 )(2:length(theta)))); % regularization

% Gradient, we multiple X' (2x12) by the computed error for each training example (12x1). Thus, the sumation is embedded in the matrix multiplication.
grad = (1/m) * (X' * ((X * theta) - y)) + ((lambda / m) .* [0; theta(2:length(theta))]);

% =========================================================================

grad = grad(:);

end
