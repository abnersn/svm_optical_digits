% Funcao que, ao passar a base e quanto se quer de representatividade da
% variancia do atributos.
function features = PCA(data, percentage)
    [~, dados, var] = pca(data(:, 1:(length(data(1, :))-1)));
    offset = 0;
    p = 0;
    while p < percentage
        offset = offset + 1;
        p = (sum(var(1:offset)) / sum(var));
    end
    fprintf('\nAs %d primeiras componentes explicam %.2f%% da variancia dos dados.\n', offset, p*100);
    features = dados(:, 1:offset);
end