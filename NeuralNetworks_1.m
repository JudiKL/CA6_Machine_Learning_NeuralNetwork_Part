%% CA6-Machine Learning Part 1: Neural Networks

% Title : Neural Networks
%
% SUMMARY: Script running the Neural Network algorithm for signal which
% have been averaged for each trial and operates the classification for the
% testing set after a training on the training set.

% Note : The following code and associated functions have been created
% based on the exercise of Coursera : Machne Learning Course (Andrew NG).
% All exercises have been fulfilled by the author's code, but fmincg that
% helps finding the minimum of cost function. All credit for general layout
% of the code goes to Coursera Machine Learning Course Exercise 3.
%
% INPUT: Matrice Data
%
% OUTPUT:  : Matrice all_theta : [class*weights]
%
% Made by: Judicaël Fassaya
% Date: May 12th, 2019


%% Inialisation
clear ; close all; clc % clear current workspace
%% Setup the parameters
lambda = 0.1;
% regularization parameter which penalized features that may have a high
% value and induce overfitting
input_layer_size  = 2;  % number of input features
hidden_layer_size = 2;   % hidden units
num_labels = 2;          % number of labels 
                          

% %% =========== Part 1: Loading and Visualizing Data =============
% %  We start the script by first loading and visualizing the dataset. 
% %
% Load Training Data :
% convert our 3D dataset into 2d matrix
% each row is a trial
% each column the average across time voltage per trial
[X,y] = extract_data_basic('01cr.fdt');
[X] = data_redux(X);
[rows, column] = size(X);
y = y';
left = find(y==1); right = find(y == 2);
m = size(X, 1);

% Load test data :
[X_test,y_test] = extract_data_basic('01fa.fdt');
[X_test] = data_redux(X_test);
y_test = y_test';
left = find(y_test==1); right = find(y_test == 2);
m = size(X_test, 1);

% Plot Examples : Training data

plot(X(left, 1), X(left, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
hold on
plot(X(right, 1), X(right, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);

xlabel('mean voltage per trial in electrode L-HEOG')
ylabel('mean voltage per trial in electrode R-HEOG')
title('distribution of ocular electrodes mean voltage as a function of electrode channel for training set')

legend('left saccadic eye movement','right saccadic eye movement')
% add legend to plotted data

fprintf('Program paused. Press enter to continue.\n');

% Plot Examples : Test data

plot(X_test(left, 1), X_test(left, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
hold on
plot(X_test(right, 1), X_test(right, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);

xlabel('mean voltage per trial in electrode L-HEOG')
ylabel('mean voltage per trial in electrode R-HEOG')
title('distribution of ocular electrodes mean voltage as a function of electrode channel for test set')

legend('left saccadic eye movement','right saccadic eye movement')

fprintf('Program paused. Press enter to continue.\n');

pause; 

%% ================ Part 2: Training : Optimize Parameters ================
% In this part, we compute neural network parameters.

% Compute the weights into variables Theta1 and Theta2

Theta1 = oneVsAll(X, y, num_labels, lambda) 
% a one vs all classification (simple logistic regression is computed in
% this case as they are two inputs) is operated to obtain the optimized weight for
% the feedforward to next layer

% intermediary
a1 = [ones(m, 1) X];
% concatenate to add bias terms to training set

z2 = a1 * Theta1'; % comute hypothesis function
a2 = sigmoid(z2); % compute sigmoid of our hypothesis to return activation
% unit value

Theta2 = oneVsAll(a2, y, num_labels, lambda) 
% Same as theta 1 : computed from activation units values of previous layer 

%% ================= Part 3: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. We now implement the "predict" function to use the
%  neural network to predict the labels of the test set. It allows
%  to compute the test set accuracy.

pred = predict(Theta1, Theta2, X_test);
predictions = pred % vector containing predicted classes

fprintf('Program paused. Press enter to continue.\n');
pause;

% Plot decision noundary
hold on
plot_x = [min(X_test(:,2))-70,  max(X_test(:,2))+70];% beginning/end of our
% decision boundary line
    
% Calculate the decision boundary line
plot_y = (Theta2(2).*plot_x + Theta2(1));
plot(plot_x, plot_y,'-')
legend('Left saccade','Right saccade','Decision boundary')
axis([-60 80 -60 80])

% This part of the code is an optionnal loop that take a random point in
% our test set and return the predicted saccade direction.
%  Randomly permute examples
rp = randperm(m); 
for i = 1:m
    % Display randomly
    fprintf('\nDisplaying ERP datas matching for direction\n');
    % set a random number in range of examples
    example = randi([1,length(X_test)])
    
    plot(X_test(example,1),X_test(example,2),'-s','MarkerSize',12)
    dataexample = [X_test(example,1),X_test(example,2)]
    % displayDatapoint on plot, precise point
    
    if y(example) == 1
        fprintf('\nCorresponding direction : left (1)\n');
    else
         fprintf('\nCorresponding direction : right (2)\n');
    end
      
    
    pred = predict(Theta1, Theta2, X_test(example,:));
    fprintf('\nNeural Network Prediction: %d', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

% Accuracy percentage comutation
correct = 0; %Initialize correct trials
%computation
for trial = 1:length(X_test) 
     if predictions(i) == y_test(i)%If prediction matches label, mark correct response
                correct = correct + 1;
     end
end

Accuracy = (correct/length(X_test))*100; %Convert to a percentage correct

   
                     

