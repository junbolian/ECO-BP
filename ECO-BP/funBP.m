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

function [fitness, net] = funBP(x, inputnum, hiddennum, outputnum, inputn, outputn, inputn_test, outputn_test)
% This function is used to calculate the fitness value
% Inputs:
%           x          input     Individual
%           inputnum   input     Number of input layer nodes
%           hiddennum  input     Number of hidden layer nodes
%           outputnum  input     Number of output layer nodes
%           net        input     Network
%           inputn     input     Training input data
%           outputn    input     Training output data
% Outputs:
%           fitness    output    Fitness value of the individual
%           net        output    Trained network

% Extract thresholds and weights
w1 = x(1:inputnum*hiddennum);
B1 = x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2 = x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2 = x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

% Assign network weights and thresholds
net = newff(minmax(inputn), [hiddennum outputnum], {'logsig', 'purelin'}, 'traingdx');
net.iw{1,1} = reshape(w1, hiddennum, inputnum);
net.lw{2,1} = reshape(w2, outputnum, hiddennum);
net.b{1} = reshape(B1, hiddennum, 1);
net.b{2} = reshape(B2, outputnum, 1);

% Set training parameters
net.trainparam.show = 50;
net.trainparam.epochs = 200;
net.trainparam.goal = 0.01;
net.trainParam.lr = 0.01;
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false;

% Train the network
net = train(net, inputn, outputn);

% Calculate MSE on the training set
Y = sim(net, inputn);
error = Y - outputn;
MSE1 = mse(error);

% Calculate MSE on the test set
Y1 = sim(net, inputn_test);
error1 = Y1 - outputn_test;
MSE2 = mse(error1);

% Calculate fitness value
fitness = MSE1 + MSE2;
end
