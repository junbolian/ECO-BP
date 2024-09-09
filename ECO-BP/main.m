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

%% BP Neural Network Prediction Optimized by Educational Competition Optimizer (ECO-BP)
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

%% Construct ECO Model
popsize = 20;           % Population size
Max_iteration = 50;     % Maximum number of iterations
lb = -5;                % Lower bound for weights and thresholds
ub = 5;                 % Upper bound for weights and thresholds
% Calculate the dimension of the problem:
% - inputnum * hiddennum: Weights from input to hidden layer
% - hiddennum * outputnum: Weights from hidden to output layer
% - hiddennum: Biases for hidden layer
% - outputnum: Biases for output layer
dim = inputnum * hiddennum + hiddennum * outputnum + hiddennum + outputnum;

% Define the objective function for optimization
fobj = @(x)funBP(x, inputnum, hiddennum, outputnum, Pn_train, Tn_train, Pn_test, Tn_test);

% Run ECO to find the best weights and thresholds for the network
[Best_pos, Best_score, ECO_cg_curve, net] = ECO(popsize, Max_iteration, lb, ub, dim, fobj);

% Plot the convergence curve of ECO
figure
plot(ECO_cg_curve, 'Color', 'r')
title('Objective Space')
xlabel('Iteration')
ylabel('Best Score Obtained So Far')
legend('ECO')
grid on;

disp('Initialized weights and biases information:')
% Test set prediction
Y = sim(net, Pn_test);
error = Y - Tn_test;
% Mean Squared Error (MSE)
E1 = mse(error);

% Plot the error distribution of ECO-BP neural network prediction
figure
plot(error, 'b:o')
title('Error Distribution of ECO-BP Neural Network Prediction')
xlabel('Index')
ylabel('Error')
grid on;

disp(['MSE obtained by ECO-BP Neural Network: ', num2str(E1)])
