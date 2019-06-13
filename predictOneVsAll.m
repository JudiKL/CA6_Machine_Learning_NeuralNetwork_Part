% Title : Predict OneVsAll
%
% SUMMARY: Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1).
%
% INPUT: X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class.
%
% OUTPUT: p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X.
% Details :  p : a vector of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 
% Made by: Judicaël Fassaya
% Date: May 12th, 2019


function p = predictOneVsAll(all_theta, X)

m = size(X, 1);
num_labels = size(all_theta, 1);

% return the following variables 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


WX = 0
SX = zeros(num_labels,1)
for i = 1:m
    p(i) = 0
    for h = 1:num_labels
        WX = X(i,:).*all_theta(h,:);    
        SX(h) = sigmoid(sum(WX,2))  
    end
    [val, index] = max(SX);
    p(i) = index;        
end
% will pick the class for which the
% corresponding logistic regression classifier outputs the highest probability and
% return the class label
% max : obtain the max for each row.     

% =========================================================================


end
