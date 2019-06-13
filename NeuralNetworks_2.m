%% CA6-Machine Learning Part 2: Neural Networks

% Title : Neural Networks adapted for 3D matrix
%
% SUMMARY: The same operations executed for Neural Network working with
% electrodes signal across time averaged are executed but adapted so that
% it operates a classification for each millisecond
%
% Note : The following code and associated functions have been created
% based on the exercise of Coursera : Machne Learning Course (Andrew NG).
% All exercises have been fulfilled by the author's code, but fmincg that
% helps finding the minimum of cost function. All credit for general layout
% of the code goes to Coursera Machine Learning Course Exercise 3.
%
% INPUT: Matrice : training data and test data
%
% OUTPUT:  : Classification across time (400 ms)
%
% Made by: Judicaël Fassaya
% Date: May 12th, 2019


%% Inialisation
clear ; close all; clc % clear current workspace
%% Setup the parameters
lambda = 1;
input_layer_size  = 2;  % number of input features
hidden_layer_size = 2;   % hidden units
num_labels = 2;          % number of labels 
                          

% %% =========== Part 1: Loading Data =============
% %  We start the exercise by first loading and visualizing the dataset. 
% %
% Load Training Data
% convert 3D dataset into 2d matrix

[X,y] = extract_data('Train.set',"saccade");
[rows, time, column] = size(X);
data_dirty = X; % help us reinitialize the data
% after each classification
y = y';
left = find(y==1); right = find(y == 2);
m = size(X, 1);
% extract time frames and store them in vector timeframes
dim = 3;
sz = size(X);
inds = repmat({1},1,ndims(X));
inds{2} = 1:sz(2);
timeframes = X(inds{:});
% timeframes = timeframes(1:2:end);
correct_total = zeros(size(timeframes));
% initialized : will store percentage correct across time
% Load Test Data
[X_test,y_test] = extract_data('Test.set',"saccade");
y_test = y_test';
data_dirty_test = X_test; % help us reinitialize the data

%% ================ Part 2: Training Neural Networks ================
% In this part, we compute neural network parameters.

% Compute the weights into variables Theta1 and Theta2
for i = 1:length(timeframes) 
    % the classification occur at each millisecond : 
    %the neural network is trained for each millisecond and tested just
    %after : it functions as the previous Neural network : but is executed
    % for each millisecond.
    [X] = (squeeze(X(:,i,:)))';
    [rows, column] = size(X);
    y = y;
    left = find(y==1); right = find(y == 2);
    m = size(X, 1);

    Theta1 = oneVsAll(X, y, num_labels, lambda); % ok

% intermediary
    a1 = [ones(m, 1) X];
% concatenate to add bias terms
%size(a1)
%size(Theta1)

    z2 = a1 * Theta1';
    a2 = sigmoid(z2); % feedforward
%
    Theta2 = oneVsAll(a2, y, num_labels, lambda);
% computed from activation units value of previous layer 

%% ================= Part 3: Implement Predict : Testing NN =================
%  After training the neural network, we would like to use it to predict
%  the labels. We now implement the "predict" function to use the
%  neural network to predict the labels of the training set. It allows
%  to compute the training set accuracy.

% Initialisation : reparing test_set at timeframe i
[X_test] = (squeeze(X_test(:,i,:)))';
    [rows, column] = size(X_test);
    left = find(y_test==1); right = find(y_test == 2);
    m = size(X_test, 1);

% prediction :
    pred = predict(Theta1, Theta2, X_test);
    predictions = pred; % vector containing matched classes predicitons

    fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_test)) * 100);

%     fprintf('Program paused. Press enter to continue.\n');
%     pause; in case of problem, you would like to pause the code there.
% to verify the accuracy

% Accuracy percentage
correct = 0; %Initialize correct trials

% computation
    for trial = 1:length(X_test) 
         if predictions(trial) == y_test(trial)%If prediction matches label, mark correct response
                    correct = correct + 1;
         end
        correct_percentage = (correct/length(X_test))*100; %Convert to a percentage
        correct_total(i) = correct_percentage; % add correct_percentage for 
    end
    X = data_dirty; % X retakes original dataset shape in order to iterate on next timeframe
    X_test = data_dirty_test; % same for data_set, allows to reintegrate the second dimension
    % of our vector. It retakes a 3D shape to iterate on the next
    % timeframe.
    
end

for t = 1:length(timeframes)
    if t == 1
        timeframes(t)=1;
    else
        timeframes(t)=t+1;
    end
end

%plotting
xlabel('Time (ms)')
ylabel('Accuracy (percentage correct)')
axis([0 400 0 100])
hold on

plot(timeframes,correct_total, '-')
legend('Classification accuracy','correct')
