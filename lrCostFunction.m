% Title : LRCOSTFUNCTION
%
% SUMMARY: Compute cost and gradient for logistic regression with 
%regularization
%
% INPUT: Dataset X, labels y, regularization parameter lambda
%
% OUTPUT : J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.]
%
% Made by: Judicaël Fassaya
% Date: May 12th, 2019

function [J, grad] = lrCostFunction(theta, X, y, lambda)
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% returns the following variables
J = 0;
grad = zeros(size(theta));

% Computation code : Fllowing code has been provided by me, 
% the layout of the loop and the initialized parameters are from coursera exercise : 

J = ( (1 / m) * sum(-y'*log(sigmoid(X*theta)) - (1-y)'*log( 1 - sigmoid(X*theta))) ) + (lambda/(2*m))*sum(theta(2:length(theta)).*theta(2:length(theta))) ;
% compute cost


% gradent descent formula : 
grad = (1 / m) * sum((sigmoid(X*theta)-y).*X);
% compute partial derivative of regularized cost function 

grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';
% regularizes all weights but bias weight
% =============================================================

grad = grad(:); % reshapes  the results in column

end
