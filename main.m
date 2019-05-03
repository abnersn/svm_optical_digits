% UNIVERSIDADE FEDERAL DO CEAR�
% T�picos em Comunica��es M�veis

% Trabalho 2 - SVM

% Abner
% �ngela
% Lucas

clear; close all; clc;
rng(1);

CLASSES = 10;
ITERACOES = 1;
CONSTANTE = 10;
KERNEL = 'linear';
PERCENTUAL_TESTE = 0.1;

% Leitura dos dados
data = csvread('training.csv');

all_features = data(:, 1:64);
% O n�mero 1 � somado �s classes para ajust�-las aos �ndices do MATLAB.
% Isso significa que o n�mero 0 corresponde � classe 1, n�mero 1 � classe 2
% e assim sucessivamente.
all_classes = data(:, 65) + 1;

% Particionamento Hold-Out
p = cvpartition(all_classes, 'HoldOut', PERCENTUAL_TESTE);

% Vetor com os erros de cada itera��o
errors = zeros(1, ITERACOES);

% Array de matrizes com o resultado esperado de cada itera��o (primeira
% coluna), ao lado do resultado obtido (segunda coluna).
results = zeros(p.TestSize, 2, ITERACOES);

for i=1:ITERACOES
    fprintf('Iteracao %d\n', i);
    
    %% Separa amostras de treino
    train_idx = training(p);
    train_features = all_features(train_idx, :);
    train_classes = all_classes(train_idx);
    
    %% Treina array de modelos SVM, um para cada classe
    models = cell(CLASSES, 1);
    for j = 1:CLASSES
        f = train_features;
        c = train_classes == j;
        count_ones = sum(uint8(c));
        while length(f) > 2 * count_ones
            r = randi([1, length(c)]);
            while c(r)
                r = randi([1, length(c)]);
            end
            f(r, :) = [];
            c(r, :) = [];
        end
        m = fitcsvm(f, uint8(c),...
            'KernelFunction', KERNEL, 'BoxConstraint', CONSTANTE,...
            'Standardize', true);
        models{j} = m;
        fprintf('- Classe %d\n', j);
    end
    
    %% Separa amostras de teste
    test_idx = test(p);
    test_features = all_features(test_idx, :);
    expected_output = all_classes(test_idx);
    
    %% Calcula predi��es
    fprintf('Calculando predi��es...\n');
    predictions = zeros(1, p.TestSize);
    for k=1:p.TestSize
        sample = test_features(k, :);
        
        % Array com as predi��es dos modelos
        model_predictions = zeros(1, CLASSES);
        for j=1:CLASSES
            [label, score] = predict(models{j}, sample);
            model_predictions(j) = score(2);
        end
        
        % O modelo com a maior predi��o � o escolhido
        [~, predicted_label] = max(model_predictions);
        predictions(k) = predicted_label;
    end
    
    results(:, 1, i) = expected_output;
    results(:, 2, i) = predictions;
    errors(i) = sum(uint8(predictions ~= expected_output'));
    
    %% Reparticiona para pr�ximo teste
    p = repartition(p);
end

%% Plota matriz de confus�o da N-�sima itera��o.
N = 1;
r = results(:, :, N);
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
plot(1:ITERACOES, errors);