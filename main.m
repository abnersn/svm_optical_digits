% UNIVERSIDADE FEDERAL DO CEAR�
% T�picos em Comunica��es M�veis

% Trabalho 2 - SVM

% Abner
% �ngela
% Lucas

clear; close all; clc;

% Leitura dos dados
data = [csvread('training.csv'); csvread('testing.csv')];

all_features = data(:, 1:64);
all_classes = data(:, 65);

p = cvpartition(all_classes, 'HoldOut');

for i=1:50
    idx = training(p);
    features = all_features(idx, :);
    p = repartition(p);
end
