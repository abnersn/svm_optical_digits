% Função responsável por deixar as classes com a mesma quantidade de amostras.
function data_ = preprocessing (data)
    n_classes = 10;
    classes = data(:, length(data(1, :))) + 1;
    n_samples = {};
    for i = 1:n_classes
        n_samples{i} = data(classes(:, 1) == i, :);
    end
    
    min_length = length(n_samples{1});
    for i = 2:length(n_samples)
        if min_length > length(n_samples{i})
            min_length = length(n_samples{i});
        end
    end
    
    for i = 1:length(n_samples)
        for j = 1:(length(n_samples{i}) - min_length)
            r = randi([1, length(n_samples{i})]);
            n_samples{1, i}(r, :) = [];
        end
    end
    
    d = [];
    for i = 1:length(n_samples)
        d(1: i * min_length, :) = [d; n_samples{1, i}(:, :)];
    end
    data_ = d;
end