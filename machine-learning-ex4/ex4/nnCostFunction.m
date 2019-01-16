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

% Adding bias feature for input layer
X = [ones(m, 1), X];

% Hidden layer propagation
hidden_layer_output = sigmoid(X*Theta1');

% Adding bias to hidden layer output
hidden_layer_output = [ones(size(hidden_layer_output, 1), 1), hidden_layer_output];

% Output layer propagation
output_layer_output = sigmoid(hidden_layer_output*Theta2');

% Transforming Y, which gives the predicted digit, into a matrix (M*nb_labels), where the right label is set to 1, and the others to 0.
y_vec = zeros(m, num_labels);
for training_example = 1:m
  y_vec(training_example, y(training_example)) = 1;
endfor

% Computing cost function
J = 1/m * sum(sum(-y_vec .* log(output_layer_output) - (1 - y_vec) .* log(1 - output_layer_output)));
J = J + (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^2 ))) * (lambda/(2*m));

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
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for training_example = 1:m
  % feedforward propagation of training example.
  input_layer = X(training_example, :)';
  
	hidden_layer_activation = Theta1 * input_layer;
	hidden_layer_output = [1; sigmoid(hidden_layer_activation)];

	output_layer_activation = Theta2 * hidden_layer_output;
	output_layer_output = sigmoid(output_layer_activation);

  % Transformingtraining example real value into logical vector.
	output_vec = ([1:num_labels] == y(training_example))';
  
	% Backward propagation of training_example
	delta_3 = output_layer_output - output_vec;
	delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(hidden_layer_activation)];
	delta_2 = delta_2(2:end); % removing bias  

	% Big delta update
	Theta1_grad = Theta1_grad + delta_2 * input_layer';
	Theta2_grad = Theta2_grad + delta_3 * hidden_layer_output';
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
