% UNIVERSIDADE FEDERAL DO CEARÁ
% Tópicos em Comunicações Móveis

% Trabalho 2 - SVM

% Abner
% Ângela
% Lucas

clear; close all; clc;

CLASSES = 10;
ITERACOES = 10;
CONSTANTE = 1;
KERNEL = 'linear';
PERCENTAGE = 0.5;

%% Importação das bases (Na base 'train' foram usados padrões de 30 pessoas e na base 'test' foram 13 pessoas diferentes)
train = csvread('training.csv');
test = csvread('testing.csv');

%% Divisão e Pré-processamento
% Usando todos os atributos da base.
train_features = train(:, 1:(length(train(1, :))-1));
test_features = test(:, 1:(length(test(1, :))-1));

% Usando PCA a fim de diminuir a quantidade de atributos, logo a complexidade.
%train_features = PCA(train, 0.8);
%test_features = PCA(test, 0.8);

% O número 1 é somado ás classes para ajustá-las aos índices do MATLAB.
% Isso significa que o número 0 da base corresponde à classe 1, o número 1 à classe 2
% e assim sucessivamente.
train_classes = train(:, length(train(1, :))) + 1;
test_classes = test(:, length(test(1, :))) + 1;

%% Particionamento da base usando a estratégia Hold-Out (Neste, usando as bases já divididas!)
p = cvpartition(test_classes, 'HoldOut', PERCENTAGE);
NUM_TEST = p.TrainSize;
% Vetor com os acertos de cada iteração
hits = zeros(1, ITERACOES);

% Array de matrizes com o resultado esperado de cada iteração (primeira
% coluna), ao lado do resultado obtido (segunda coluna).
results = zeros(NUM_TEST, 2, ITERACOES);

%% TREINAMENTO E TESTE
%% Treina array de modelos SVM, um para cada classe (1 vs ALL)
    models = cell(CLASSES, 1);
    for j = 1:CLASSES
        models{j} = fitcsvm(train_features, uint8(train_classes == j)*j,...
            'KernelFunction', KERNEL, 'BoxConstraint', CONSTANTE,...
            'Standardize', true, 'ClassNames', {int2str(0), int2str(j)});
        fprintf('- Classe %d\n', j);
    end

for i=1:ITERACOES
    fprintf('Iteração %d\n', i);
    
    %% Separa amostras de teste
    test_idx = training(p);
    features = test_features(test_idx, :);
    classes = test_classes(test_idx);
    
    %% Calcula predições
    fprintf('Calculando predições...\n');    
    % Array com as predições dos modelos
    model_predictions = zeros(NUM_TEST, CLASSES);
    
    for j=1:CLASSES
        [label, score] = predict(models{j}, features);
        model_predictions(:, j) = score(:, 2);
    end
        
    % O modelo com a maior predição é o escolhido
    [~, predictions] = max(model_predictions, [], 2);
    
    % Guarda os resultados de cada iteração e soma a quantidade de acertos
    results(:, 1, i) = classes;
    results(:, 2, i) = predictions;
    hits(i) = sum(uint8(predictions == classes));
    
    %% Reparticiona para próximo teste
    p = repartition(p);
end

%% Plota matriz de confusão média.
r = floor(mean(results, ITERACOES));
targets = zeros(CLASSES, NUM_TEST);
outputs = zeros(CLASSES, NUM_TEST);
subs = 1:NUM_TEST;
targets_idx = sub2ind(size(targets), r(:, 1), subs');
outputs_idx = sub2ind(size(outputs), r(:, 2), subs');
targets(targets_idx) = 1;
outputs(outputs_idx) = 1;
plotconfusion(targets, outputs);

%% Plota erros
figure;
accuracy = hits * 100 / NUM_TEST;
plot(1:ITERACOES, accuracy, 'bo--');
hold on;
x = linspace(1, ITERACOES);
plot(x , mean(accuracy) * ones(1, length(x)), 'm-')
hold off;
legend('Taxa de acertos por iteração.', "Taxa de acerto média. (" + mean(accuracy) + "%)", 'Location', 'southoutside');
title("Taxa de acertos a cada iteração (" + NUM_TEST + " amostras de teste).");
