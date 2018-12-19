function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Adding bias feature for input layer
X = [ones(m, 1), X];

% Hidden layer propagation
hidden_layer_output = sigmoid(X*Theta1');

% Adding bias to hidden layer output
hidden_layer_output = [ones(size(hidden_layer_output, 1), 1), hidden_layer_output];

% Output layer propagation
output_layer_output = sigmoid(hidden_layer_output*Theta2');

% Getting best hypotesis for all examples
[~, p] = max(output_layer_output, [], 2);


% Theta1       : Matrix with activation nodes as rows, and input feature weights as columns. The first columns being the bias weight.
% X            : Matrix with training examples as rows, and features as columns (digit pixels grayscale values). We added the first column as the bias feature (1).
%              -> sigmoid(X*Theta1'): X*Theta' gives a Matrix with training examples (digits) as rows, and hidden layer activation nodes' outputs (which are sigmoid(θ * X), where θ is node weights and X digit features) as columns.
%              -> output_layer_output: Matrix with training examples as rows, and activation nodes of the output layer's outputs as columns.

% =========================================================================
end
