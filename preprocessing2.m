% Função responsável por remover aleatóriamente dados não pertecente a
% classe que está sendo atualmente modelada no 1 vs ALL, até que fique 
% no mínimo 1/3 das amostras correspondendo a classe atualmente modelada.
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