% UNIVERSIDADE FEDERAL DO CEARÁ
% Tópicos em Comunicações Móveis

% Trabalho 2 - SVM

% Abner
% Ângela
% Lucas

clear; close all; clc;
rng(1);

CLASSES = 10;
ITERACOES = 50;
CONSTANTE = 1;
KERNEL = 'linear';
PERCENTUAL_TESTE = 0.3;

%% Importação da base
data = csvread('training.csv'); %csvread('testing.csv')];

%data = preprocessing(data);

all_features = data(:, 1:64);
% O número 1 é somado ás classes para ajustá-las aos índices do MATLAB.
% Isso significa que o número 0 da base corresponde à classe 1, o número 1 à classe 2
% e assim sucessivamente.
all_classes = data(:, 65) + 1;

%% Particionamento da base usando a estratégia Hold-Out
p = cvpartition(all_classes, 'HoldOut', PERCENTUAL_TESTE);

% Vetor com os acertos de cada iteração
hits = zeros(1, ITERACOES);

% Array de matrizes com o resultado esperado de cada iteração (primeira
% coluna), ao lado do resultado obtido (segunda coluna).
results = zeros(p.TestSize, 2, ITERACOES);

for i=1:ITERACOES
    fprintf('Iteração %d\n', i);
    
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
    
    %% Calcula predições
    fprintf('Calculando predições...\n');    
    % Array com as predições dos modelos
    model_predictions = zeros(p.TestSize, CLASSES);
    
    for j=1:CLASSES
        [label, score] = predict(models{j}, test_features);
        model_predictions(:, j) = score(:, 2);
    end
        
    % O modelo com a maior predição é o escolhido
    [~, predictions] = max(model_predictions, [], 2);
    
    % Guarda os resultados de cada iteração e soma a quantidade de acertos
    results(:, 1, i) = expected_output;
    results(:, 2, i) = predictions;
    hits(i) = sum(uint8(predictions == expected_output));
    
    %% Reparticiona para próximo teste
    p = repartition(p);
end

%% Plota matriz de confusão média.
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
plot(1:ITERACOES, hits, 'bo--');
hold on;
x = linspace(1, ITERACOES);
plot(x , mean(hits) * ones(1, length(x)), 'm-')
hold off;
acuracy = mean(hits) * 100 / p.TestSize;
legend('Acertos por Iteração.', "Taxa de acerto média. (" + acuracy + "%)", 'Location', 'southoutside');
title("Número de acertos a cada iteração (" + p.TestSize + " amostras de teste).");
