% UNIVERSIDADE FEDERAL DO CEARA
% Topicos em Comunicacoes Moveis

% Trabalho 2 - SVM

% Abner
% Angela
% Lucas

clear; close all force; clc;
rng(1); % Para reproduzir os resultados

%% Configuracoes
CLASSES = 10;
ATRIBUTOS = 64;
ITERACOES = 50;
CONSTANTE_SVM = 1;
KERNEL = 'linear';
USANDO_PCA = false;
VARIANCIA_PCA = 0.8;
PERCENTUAL_TESTE = 0.3;

fprintf('*** Execucao do SVM sobre a base Optical Digits (https://bit.ly/2xDW3IY) ***\n\n');
fprintf('Configuracoes da base:\n');
fprintf('- Classes: %d\n', CLASSES);
fprintf('- Atributos: %d\n', ATRIBUTOS);
fprintf('- Iteracoes: %d\n', ITERACOES);
if USANDO_PCA
    fprintf('- Variancia minima PCA: %.2f%%\n', VARIANCIA_PCA * 100);
end
fprintf('\nConfiguracoes de particionamento:\n');
fprintf('- Metodo: Hold Out\n');
fprintf('- Percentual para teste: %.2f%%\n', PERCENTUAL_TESTE * 100);
fprintf('\nConfiguracoes do modelo:\n');
fprintf('- Constante de penalidade: %d\n', CONSTANTE_SVM);
fprintf('- Kernel: %s\n', KERNEL);

%% Importacao e processamento inicial da base
data = csvread('training.csv'); %csvread('testing.csv')];

% O numero 1 sera somado as classes para ajusta-las aos indices do MATLAB.
% Isso significa que o numero 0 da base corresponde a classe 1, o numero 1
% a classe 2, e assim sucessivamente.
all_classes = data(:, size(data, 2)) + 1;

% Usando PCA a fim de diminuir a quantidade de atributos, logo a complexidade.
if USANDO_PCA
    all_features = PCA(data(:, 1:ATRIBUTOS), VARIANCIA_PCA);
else
    all_features = data(:, 1:ATRIBUTOS);
end

% Barra de progresso
w = waitbar(0,'Iteracao 1, Classe 0','Name','Treinando modelos...',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
fprintf('-------------\n');

%% Particionamento da base usando a estrategia Hold-Out
p = cvpartition(all_classes, 'HoldOut', PERCENTUAL_TESTE);

%% Armazenagem dos resultados
% Vetor para armazenar os acertos de cada iteracao
hits = zeros(1, ITERACOES);

% Vetor para armazenar o tempo em cada iteracao
times = zeros(1, ITERACOES);

% Array de matrizes com o resultado esperado de cada iteracao (primeira
% coluna), ao lado do resultado obtido (segunda coluna).
results = zeros(p.TestSize, 2, ITERACOES);

%% Loop principal
tic;
for i=1:ITERACOES
    t = tic;
    
    % Verifica se botao de cancelar foi pressionado
    if getappdata(w,'canceling')
        i = i - 1;
        break
    end
    
    %% Separa amostras de treino
    train_idx = training(p);
    train_features = all_features(train_idx, :);
    train_classes = all_classes(train_idx);
    
    %% Treina array de modelos SVM, um para cada classe (1 vs ALL)
    models = cell(CLASSES, 1);
    for j = 1:CLASSES
        c = train_classes == j;
        models{j} = fitcsvm(train_features, uint8(c)*j,...
            'KernelFunction', KERNEL, 'BoxConstraint', CONSTANTE_SVM,...
            'Standardize', true, 'ClassNames', {int2str(0), int2str(j)});
        progress = (i - 1 + (j/CLASSES)) / ITERACOES;
        waitbar(progress , w, sprintf('Iteracao %d - Numero %d - (%.2f%%)', i, j - 1, progress*100))
    end
    
    %% Separa amostras de teste
    test_idx = test(p);
    test_features = all_features(test_idx, :);
    expected_output = all_classes(test_idx);
    
    %% Calcula predicoes
    
    % Array com as predicoes dos modelos
    model_predictions = zeros(p.TestSize, CLASSES);
    
    for j=1:CLASSES
        [label, score] = predict(models{j}, test_features);
        model_predictions(:, j) = score(:, 2);
    end
        
    % O modelo com a maior predicao sera o escolhido
    [~, predictions] = max(model_predictions, [], 2);
    
    % Guarda os resultados de cada iteracao e soma a quantidade de acertos
    results(:, 1, i) = expected_output;
    results(:, 2, i) = predictions;
    hits(i) = sum(uint8(predictions == expected_output));
    
    %% Reparticiona para proximo teste
    p = repartition(p);
    
    %% Tempo de execucao
    times(i) = toc(t);
end

delete(w);

%% Plota grafico 3D das 3 primeiras componentes principais
visualizacao_pca(data(:, 1:ATRIBUTOS), all_classes, CLASSES);

%% Plota acertos das iteracoes executadas
figure;
subplot(1, 2, 1);
accuracy = hits * 100 / p.TestSize;
accuracy = accuracy(1:i);
plot(1:i, accuracy, 'bo--');
hold on;
x = linspace(1, i);
plot(x , mean(accuracy) * ones(1, length(x)), 'm-')
hold off;
legend('Taxa de acertos por iteracao.', "Taxa de acerto media. (" + mean(accuracy) + "%)", 'Location', 'southoutside');
title("Taxa de acertos a cada iteracao (" + p.TestSize + " amostras de teste).");
xlabel('Iteracao');
ylabel('Acertos (%)');


%% Plota tempo das iteracoes executadas
%figure;
subplot(1, 2, 2);
times = times(1:i);
plot(1:i, times, 'bo--');
hold on;
x = linspace(1, i);
plot(x , mean(times) * ones(1, length(x)), 'm-')
hold off;
legend('Tempo de execucao por iteracao.', "Tempo medio. (" + mean(times) + ")", 'Location', 'southoutside');
mins = floor(sum(times) / 60);
segs = floor(mod(sum(times), 60));
title("Tempo de execucao (" + mins + "min e " + segs + "s).");
xlabel('Iteracao');
ylabel('Tempo (s)');


%% Plota matriz de confusao media.
figure;
r = floor(mean(results, ITERACOES));
targets = zeros(CLASSES, p.TestSize);
outputs = zeros(CLASSES, p.TestSize);
subs = 1:p.TestSize;
targets_idx = sub2ind(size(targets), r(:, 1), subs');
outputs_idx = sub2ind(size(outputs), r(:, 2), subs');
targets(targets_idx) = 1;
outputs(outputs_idx) = 1;
plotconfusion(targets, outputs);
