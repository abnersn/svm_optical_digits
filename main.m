% UNIVERSIDADE FEDERAL DO CEARÁ
% Tópicos em Comunicações Móveis

% Trabalho 2 - SVM

% Abner
% Ângela
% Lucas

clear; close all; clc;

CLASSES = 10;
ITERACOES = 50;
CONSTANTE = 1;
KERNEL = 'gaussian';

% Leitura dos dados
data = [csvread('training.csv')];

all_features = data(:, 1:64);
all_classes = data(:, 65);

p = cvpartition(all_classes, 'HoldOut');

for i=1:ITERACOES
    fprintf('Iteracao %d\n', i);
    
    %% Separa amostras de treino
    train_idx = training(p);
    train_features = all_features(train_idx, :);
    train_classes = all_classes(train_idx);
    
    %% Treina array de modelos SVM, um para cada classe
    models = {};
    for j = 1:CLASSES
        c = uint8(train_classes == j - 1);
        m = fitcsvm(train_features, c, 'KernelFunction', KERNEL, 'BoxConstraint', CONSTANTE);
        models{j} = m;
        fprintf('- Classe %d\n', j - 1);
    end
    
    %% Separa amostras de teste
    test_idx = test(p);
    test_features = all_features(test_idx, :);
    test_classes = all_classes(test_idx);
    
    %% Calcula predições
    fprintf('Calculando predições...\n');
    predictions = zeros(size(test_classes));
    for k=1:size(test_features, 1)
        sample = test_features(k, :);
        
        % Array com as predições dos modelos
        model_predictions = zeros(1, CLASSES);
        for j=1:CLASSES
            [label, score] = predict(models{j}, sample);
            if label == 1
                model_predictions(j) = max(score);
            end
        end
        
        % O modelo com a maior predição é o escolhido
        [~, predicted_label] = max(model_predictions);
        predictions(k) = predicted_label - 1;
    end
    
    %% Plota matriz de confusão
    plotconfusion(test_classes, predictions);
    
    %% Reparticiona para próximo teste
    p = repartition(p);
end
