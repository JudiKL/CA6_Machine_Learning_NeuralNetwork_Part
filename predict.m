function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
% Additional info : make predictions using
% the learned learned neural network.  p is a 
% vector containing labels between 1 to num_labels.

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
direction = 'undefined';
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== My CODE HERE ======================
% All credit for the layout of the loop goes to coursera. Following
% code has been written by me.
%
%

a1 = [ones(m, 1) X];
% concatenate to add bias terms
%size(a1)
%size(Theta1)

z2 = a1 * Theta1';
a2 = sigmoid(z2);
% feedforward from input to hidden layer
%size(a2)

a2 = [ones(size(a2,1), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
% feedforward from hidden to output layer
%size(a3) = probability for each class gathered
%

[val, index] = max(a3,[],2);
p = index;
% allows to return p based on the highest probabiity 


% =========================================================================


end
