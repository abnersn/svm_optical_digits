clear; close all; clc;
QTD = 4;

data = csvread('training.csv');
r = randi(length(data), QTD, 1);

for j=1:QTD
    img = zeros(8);
    for i=1:8:63
        img(floor(i / 8) + 1, :) = data(r(j), i:i+7);
    end
    figure;
    %img = 2 * img / max(max(img));%Descomente para binarizar as imagens
    image(img);
    title("Classe = " + data(r(j), 65));
end