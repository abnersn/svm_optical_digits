function visualizacao_pca(samples, classes, num_classes)
%VISUALIZACAO_PCA Exibe grafico 3D com as 3 primeiras componentes
%principais da base de dados.

[~, componentes, var] = pca(samples);
componentes = componentes(:, 1:3);
var = sum(var(1:3)) * 100 / sum(var);

colors = {[.8 0 0] [0 .8 0] [0 0 .8] [.8 .8 0] [.8 0 .8] [0 .8 .8] [0 0 0] [.3 .3 .3] [.5 .5 .5] [.8 .8 .8]};
figure;
hold on;
for i = 1:num_classes
    indexes = classes == i;
    plot_samples = componentes(indexes, :);
    scatter3(plot_samples(:, 1), plot_samples(:, 2), plot_samples(:, 3),...
        'filled', 'MarkerFaceColor', colors{1, i});
end

lgd = strings(num_classes, 1);
for i = 1:num_classes
    lgd(i) = sprintf('Numero %d', i - 1);
end

grid on;
legend(lgd, 'Location', 'eastoutside');
title("Visualizacao das 3 primeiras componentes principais ("+var+"%)");
xlabel('1o Componente');
ylabel('2o Componente');
zlabel('3o Componente');
view(45, 45);

hold off;

end

