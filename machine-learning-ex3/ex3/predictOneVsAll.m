function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%

[~, p] = max(sigmoid(X*all_theta'), [], 2);

% all_theta: Matrix with our labels as lines, and our parameters for each features (the N pixels of a digit) as columns
% X        : Matrix with our training example as rows (digits) and features (the N pixels of a digit) as columns
%          -> X*all_theta'                     : Matrix with training examples as rows (digits) and sum of linear function θx for each label as columns. 
%          -> sigmoid(X*all_theta')            : Matrix with training examples as rows (digits) and hypotesis computation for each label as columns.
%          -> max(sigmoid(X*all_theta'), [], 2): Gives two lists of N values, which are the best hypotesis probability, and the id of its columns, for each training example. We only take the id, because it is the best matching label. 

% =========================================================================
end
