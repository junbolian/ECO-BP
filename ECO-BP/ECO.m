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


function [Best_score,Best_pos, curve, BestNet] = ECO(N, Max_iter, lb, ub, dim, fobj)

H = 0.5;  % Learning habit boundary
G1 = 0.2; % proportion of primary school (Stage 1)
G2 = 0.1; % proportion of middle school (Stage 2)

% BestF: Best value in a certain iteration
% WorstF: Worst value in a certain iteration
% GBestF: Global best fitness value
% AveF: Average value in each iteration

G1Number = round(N * G1); % Number of G1
G2Number = round(N * G2); % Number of G2
if (max(size(ub)) == 1)
    ub = ub .* ones(1, dim);
    lb = lb .* ones(1, dim);
end
net = {};
net0 = {};
%% Initialization
X0 = initializationLogistic(N, dim, ub, lb); %Logistic Chaos Mapping Initialization
X = X0;

% Compute initial fitness values
fitness = zeros(1, N);
for i = 1:N
    [fitness(i),net0{i}] = fobj(X(i, :));
end
[fitness, index] = sort(fitness); % sort
GBestF = fitness(1); % Global best fitness value
AveF = mean(fitness);
for i = 1:N
    X(i, :) = X0(index(i), :);
    net{i} = net0{index(i)};
end
curve = zeros(1, Max_iter);
avg_fitness_curve = zeros(1, Max_iter);
GBestX = X(1, :); % Global best position
X_new = X;
BestNet = net{1};
search_history = zeros(N, Max_iter, dim);
fitness_history = zeros(N, Max_iter);

%% Start search
for i = 1:Max_iter
    if mod(i,100) == 0
      display(['At iteration ', num2str(i), ' the fitness is ', num2str(curve(i-1))]);
    end
    avg_fitness_curve(i) = AveF;
    R1 = rand(1);
    R2 = rand(1);
    P = 4 * randn * (1 - i / Max_iter);
    E = (pi * i) / (P * Max_iter);
    w = 0.1 * log(2 - (i / Max_iter));
    for j = 1:N
        % Stage 1:Primary school competition
        if mod(i, 3) == 1
            if j >= 1 && j <= G1Number %Primary school Site Selection Strategies
                X_new(j, :) = X(j, :) + w * (mean(X(j, :)) - X(j, :)) .* Levy(dim); 
            else %Competitive Strategies for Students (Stage 1)
                X_new(j, :) = X(j, :) + w * (close(X(j, :),1,X) - X(j, :)) .* randn(1);
            end
        % Stage 2:Middle school competition
        elseif mod(i, 3) == 2
            if j >= 1 && j <= G2Number %Middle school Site Selection Strategies
                X_new(j, :) = X(j, :) + (GBestX - mean(X)) * exp(i / Max_iter - 1) .* Levy(dim);
            else %Competitive Strategies for Students (Stage 2)
                if (R1 < H)
                    X_new(j, :) = X(j, :) - w * close(X(j, :),2,X) - P * (E * w *close(X(j, :),2,X) - X(j, :));
                else
                    X_new(j, :) = X(j, :) - w * close(X(j, :),2,X) - P * (w * close(X(j, :),2,X) - X(j, :));
                end
            end
        % Stage 3:High school competition
        else
            if j >= 1 && j <= G2Number %High school Site Selection Strategies
                X_new(j, :) = X(j, :) + (GBestX - X(j, :)) *randn - (GBestX - X(j, :)) * randn;
            else %Competitive Strategies for Students (Stage 3)
                if (R2 < H)
                    X_new(j, :) = GBestX - P * (E * GBestX - X(j, :));
                else
                    X_new(j, :) = GBestX - P * (GBestX - X(j, :));
                end
            end
        end

        % Boundary control
        Flag4ub=X_new(j,:)>ub;
        Flag4lb=X_new(j,:)<lb;
        X_new(j,:)=(X_new(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
       
        % Finding the best location so far
        fitness_new(j) = fobj(X_new(j, :));
        if fitness_new(j) >  fitness(j)
            fitness_new(j) = fitness(j);
            X_new(j, :) = X(j,:);
        end

        if fitness_new(j) < GBestF
            GBestF = fitness_new(j);
            GBestX = X_new(j, :);
            BestNet = net{j};

        end
    end

    X = X_new;
    fitness = fitness_new;  
    curve(i) = GBestF;
    Best_pos = GBestX;
    Best_score = curve(end);
    search_history(:, i, :) = X;
    fitness_history(:, i) = fitness;
    
    % Sorting and updating
    [fitness, index] = sort(fitness); % sort
    for j = 1:N
        X0(j, :) = X(index(j), :);
        net0{j}=net{index(j)};
    end 
    X = X0;
    net = net0;
end

%%  Levy search strategy
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) *sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / beta);
    o = step;
end    

%%  Choosing the Nearest School
function o = close(t,G,X)
  m = X(1,:);
  if G == 1
     for s = 1:G1Number
        school = X(s, :);
        if  sum(abs(m - t)) > sum(abs(school - t))
            m = school;
        end
     end
  else
     for s = 1:G2Number
        school = X(s, :);
        if  sum(abs(m - t)) > sum(abs(school - t))
            m = school;
        end
     end
  end
  o = m;  
end

end