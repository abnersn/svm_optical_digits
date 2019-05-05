function visualizacao_pca(samples, classes, num_classes)
%VISUALIZACAO_PCA Exibe grafico 3D com as 3 primeiras componentes
%principais da base de dados.

[~, componentes] = pca(samples);
componentes = componentes(:, 1:3);

figure;
hold on;
for i = 1:num_classes
    indexes = classes == i;
    plot_samples = componentes(indexes, :);
    scatter3(plot_samples(:, 1), plot_samples(:, 2), plot_samples(:, 3),...
        'filled');
end

lgd = strings(num_classes, 1);
for i = 1:num_classes
    lgd(i) = sprintf('Caractere %d', i - 1);
end

grid on;
legend(lgd);
title('Visualizacao das 3 primeiras componentes principais');
xlabel('1ª Componente');
ylabel('2ª Componente');
zlabel('3ª Componente');
view(45, 45);

hold off;

end

