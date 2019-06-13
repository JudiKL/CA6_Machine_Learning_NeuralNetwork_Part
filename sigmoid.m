% Title : Sigmoid
%
% SUMMARY: %SIGMOID Compute sigmoid functoon
%
% INPUT: Matrice z
%
% OUTPUT:  : Matrice containing sigmoid transformations of all marix
% elements. 
% g = SIGMOID(z) computes the sigmoid of z.
%
% Made by: Judicaël Fassaya
% Date: May st, 2019



function g = sigmoid(z)


g = 1.0 ./ (1.0 + exp(-z));
end
