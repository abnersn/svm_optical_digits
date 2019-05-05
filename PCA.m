% Função que, ao passar a base e quanto se quer de representatividade da
% variância do atributos.
function features = PCA(data, percentage)
    [~, dados, var] = pca(data(:, 1:(length(data(1, :))-1)));
    offset = 1;
    p = 0;
    while p < percentage
        p = (sum(var(1:offset)) / sum(var));
        offset = offset + 1;
    end
    fprintf('As %d primeiras componentes explicam %.2f %% da variância dos dados.\n', offset, (sum(var(1:offset)) / sum(var))*100);
    features = dados(:, 1:offset);
end