%___________________________________________________________________________________________________________________________________________________%

% 📜 ECO-BP source codes (version 1.0)

% 🌐 Website and codes of ECO-BP: https://github.com/junbolian/ECO-LSTM

% 📅 Last update: Sep 09 2024

% 📧 E-Mail: junbolian@qq.com
  
%----------------------------------------------------------------------------------------------------------------------------------------------------%

% 👥 Corresponding Author: Junbo Lian (junbolian@qq.com)

%----------------------------------------------------------------------------------------------------------------------------------------------------%

% 📊 You can use and compare with other optimization methods developed by the same authors:

% 📜 Educational Competition Optimizer (ECO) - 2024: 
% 🔗 http://www.aliasgharheidari.com/ECO.html OR https://github.com/junbolian/ECO

% 📜 Parrot Optimizer (PO) - 2024: 
% 🔗 http://www.aliasgharheidari.com/PO.html OR https://github.com/junbolian/ECO

% 📜 Human Evolutionary Optimization Algorithm (HEOA) - 2023: 
% 🔗 https://github.com/junbolian/HEOA

%____________________________________________________________________________________________________________________________________________________%
%% Basic BP Neural Network Prediction
clear all 
clc
warning off;

%% Load Data
load data
% Training set: 1900 samples
P_train = input(1:1900, :)';
T_train = output(1:1900);
% Test set: 100 samples
P_test = input(1901:2000, :)';
T_test = output(1901:2000);

%% Normalize Data
% Training set normalization
[Pn_train, inputps] = mapminmax(P_train, -1, 1);
Pn_test = mapminmax('apply', P_test, inputps);
% Test set normalization
[Tn_train, outputps] = mapminmax(T_train, -1, 1);
Tn_test = mapminmax('apply', T_test, outputps);

%% Construct Neural Network Structure
% Create Neural Network
inputnum = 2;      % Number of input layer nodes (2-dimensional features)
hiddennum = 10;    % Number of hidden layer nodes
outputnum = 1;     % Number of output layer nodes

% Define the neural network with one hidden layer
net = newff(minmax(Pn_train), [hiddennum outputnum], {'logsig' 'purelin'}, 'traingdx');

% Set training parameters
net.trainparam.show = 50;      % Show progress every 50 iterations
net.trainparam.epochs = 200;   % Maximum number of epochs
net.trainparam.goal = 0.01;    % Training goal (mean squared error)
net.trainParam.lr = 0.01;      % Learning rate

% Train the network
net = train(net, Pn_train, Tn_train);

%% Test the Neural Network
% Test set prediction
Y = sim(net, Pn_test);
error = Y - Tn_test;

% Calculate Mean Squared Error (MSE)
E1 = mse(error);

% Plot the error distribution
figure
plot(error, 'b:o')
title('Error Distribution of BP Neural Network Prediction')
xlabel('Index')
ylabel('Error')
grid on

% Display the MSE
disp(['MSE obtained by BP Neural Network: ', num2str(E1)])
