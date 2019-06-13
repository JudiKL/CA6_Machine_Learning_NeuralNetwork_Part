% TITLE: Reduces data dimensionality after extraction
%
% SUMMARY: This script reduces the EEG data third dimension in order to
% match the label output of the function 'Extract EEG data into Matlab
% Matrix'
%
% INPUT: Name of data argument of the function 'Extract EEG data'
%
% OUTPUT: 2*118 matrix containing voltage across time averages for each trial for each
% eye movement-electrode
%
% Made by: Judicaël Fassaya
% Data: May 21st, 2019

% Dimensionality reduction :
% this script aims at averaging time dimensionality of the dataset so as to
% end up with a 2D matrix

% to refine 
function [data] = data_redux(data)
% 1) averaging
data = mean(data,2)
% 2) Dimensionality reduction : 
data = squeeze(data)
data = data'
end

% % Applicated example :
% [output_matrix,label] = extract_data('01cr.set')
% % 1) averaging
% av_matrix = mean(output_matrix,2)
% % 2) Dimensionality reduction : 
% two_dmatrix = squeeze(av_matrix)
% two_dmatrix = two_dmatrix'