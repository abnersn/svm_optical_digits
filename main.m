% UNIVERSIDADE FEDERAL DO CEARÁ
% Tópicos em Comunicações Móveis

% Trabalho 2 - SVM

% Abner
% Ângela
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
