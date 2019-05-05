% Funcao responsavel por remover aleatoriamente dados nao pertecente a
% classe que esta sendo atualmente modelada no 1 vs ALL, ate que fique 
% no minimo 1/3 das amostras correspondendo a classe atualmente modelada.
function [f_, c_] = preprocessing2 (f, c)
    count_ones = sum(uint8(c));
    while length(f) > 2 * count_ones
        r = randi([1, length(c)]);
        while c(r)
            r = randi([1, length(c)]);
        end
        f(r, :) = [];
        c(r, :) = [];
    end
    f_ = f;
    c_ = c;
end