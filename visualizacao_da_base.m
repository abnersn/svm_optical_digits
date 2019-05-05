clear; close all; clc;
CLASSES = 10;

data = csvread('training.csv');
base = {};
for i = 1:CLASSES
    base{i} = data(data(:, 65) == (i-1), :);
end

r = randi(length(base{:, 1}), CLASSES, 1);

for i=1:CLASSES
    img = zeros(8);
    for j=1:8:63
        img(floor(j / 8) + 1, :) = base{i}(r(i), j:j+7);
    end
    figure;
    %img = 2 * img / max(max(img));%Descomente para binarizar as imagens
    img = image(img);
    title("Classe = " + base{i}(r(i), 65));
    
    waitfor(img);
    %waitforbuttonpress
end