% Title : oneVsAll
%
% SUMMARY: Implement Logistic Regression to One-vs-all classification
% Details : [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i
%
% INPUT: Matrice : [all examples*all features]
%
% OUTPUT:  : Matrice all_theta : [labels*weights]
%
% Made by: Judicaël Fassaya
% Date: May 12th, 2019


function [all_theta] = oneVsAll(X, y, num_labels, lambda)

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% return the following initialized variable 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
%
% Note: fmincg optimizes the cost function and will return theta and the cost 
% fmincg works similarly to fminunc, but is more efficient when we
% are dealing with large number of parameters.

% Following code is provided by me, initialized parameters have been done 
% by the exercise provider as well as the layout of the loop.
 % Set Initial theta
initial_theta = zeros(n + 1, 1);
 % Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Run fmincg to obtain the optimal theta
for c = 1:num_labels
all_theta(c,:) = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
% a for-loop (for c = 1:num_labels) to loop over the different classes.
% output : optimal thetas for each row that is to say each class

% y == c allows to obtain a vector of 1's and 0's that tell you whether
% the ground truth is true/false for this class.






% =========================================================================


end
