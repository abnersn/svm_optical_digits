% UNIVERSIDADE FEDERAL DO CEARÁ
% Tópicos em Sistemas de Comunicações Móveis

% Trabalho 2 - SVM

pkg load statistics;

clear; close all; clc;

% Leitura dos dados
data = [csvread('training.csv'); csvread('testing.csv')];

features = data(:, 1:64);
classes = data(:, 65);

p = cvpartition(classes, 'HoldOut')

for i=1:50
  % Adicionar códigos aqui...
  
  p = repartition(p)
endfor
