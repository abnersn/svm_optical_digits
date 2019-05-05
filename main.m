% UNIVERSIDADE FEDERAL DO CEARA
% Topicos em Comunicacoes Moveis

% Trabalho 2 - SVM

% Abner
% Angela
% Lucas

clear; close all; clc;
rng(1);

CLASSES = 10;
ITERACOES = 50;
CONSTANTE = 1;
KERNEL = 'linear';
PERCENTUAL_TESTE = 0.3;

%% Importacao da base
data = csvread('training.csv'); %csvread('testing.csv')];

%% Pre processamento
%data = preprocessing(data);

% Binarizacao da base, pre-processamento opcional
%data(:, 1:(length(data(1, :))-1)) = (data(:, 1:(length(data(1, :))-1)) * 2 / max(max(data))) >= 1;

% Usando todos os atributos da base.
all_features = data(:, 1:(length(data(1, :))-1));

% Usando PCA a fim de diminuir a quantidade de atributos, logo a complexidade.
% all_features = PCA(data, 0.8);

% O numero 1 e somado as classes para ajusta-las aos indices do MATLAB.
% Isso significa que o numero 0 da base corresponde a classe 1, o numero 1 a classe 2
% e assim sucessivamente.
all_classes = data(:, length(data(1, :))) + 1;

%% Particionamento da base usando a estrategia Hold-Out
p = cvpartition(all_classes, 'HoldOut', PERCENTUAL_TESTE);

% Vetor com os acertos de cada iteracao
hits = zeros(1, ITERACOES);

% Array de matrizes com o resultado esperado de cada iteracao (primeira
% coluna), ao lado do resultado obtido (segunda coluna).
results = zeros(p.TestSize, 2, ITERACOES);

for i=1:ITERACOES
    fprintf('Iteracao %d\n', i);
    
    %% Separa amostras de treino
    train_idx = training(p);
    train_features = all_features(train_idx, :);
    train_classes = all_classes(train_idx);
    
    %% Treina array de modelos SVM, um para cada classe (1 vs ALL)
    models = cell(CLASSES, 1);
    for j = 1:CLASSES
        f = train_features;
        c = train_classes == j;
        %[f ,c] = preprocessing2(f, c);
        models{j} = fitcsvm(f, uint8(c)*j,...
            'KernelFunction', KERNEL, 'BoxConstraint', CONSTANTE,...
            'Standardize', true, 'ClassNames', {int2str(0), int2str(j)});
        fprintf('- Classe %d\n', j);
    end
    
    %% Separa amostras de teste
    test_idx = test(p);
    test_features = all_features(test_idx, :);
    expected_output = all_classes(test_idx);
    
    %% Calcula predicoes
    fprintf('Calculando predicoes...\n');    
    % Array com as predicoes dos modelos
    model_predictions = zeros(p.TestSize, CLASSES);
    
    for j=1:CLASSES
        [label, score] = predict(models{j}, test_features);
        model_predictions(:, j) = score(:, 2);
    end
        
    % O modelo com a maior predicao e o escolhido
    [~, predictions] = max(model_predictions, [], 2);
    
    % Guarda os resultados de cada iteracao e soma a quantidade de acertos
    results(:, 1, i) = expected_output;
    results(:, 2, i) = predictions;
    hits(i) = sum(uint8(predictions == expected_output));
    
    %% Reparticiona para proximo teste
    p = repartition(p);
end

%% Plota matriz de confusao media.
r = floor(mean(results, ITERACOES));
targets = zeros(CLASSES, p.TestSize);
outputs = zeros(CLASSES, p.TestSize);
subs = 1:p.TestSize;
targets_idx = sub2ind(size(targets), r(:, 1), subs');
outputs_idx = sub2ind(size(outputs), r(:, 2), subs');
targets(targets_idx) = 1;
outputs(outputs_idx) = 1;
plotconfusion(targets, outputs);

%% Plota erros
figure;
accuracy = hits * 100 / p.TestSize;
plot(1:ITERACOES, accuracy, 'bo--');
hold on;
x = linspace(1, ITERACOES);
plot(x , mean(accuracy) * ones(1, length(x)), 'm-')
hold off;
legend('Taxa de acertos por iteracao.', "Taxa de acerto media. (" + mean(accuracy) + "%)", 'Location', 'southoutside');
title("Taxa de acertos a cada iteracao (" + p.TestSize + " amostras de teste).");
