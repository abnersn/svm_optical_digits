# Classificador SVM para a base Optical Digits

Link para o dataset:

```
https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
```

### Base de dados
* **64** Atributos **(0..16)** e **1** para classe **(0..9)**
* **5620** amostras (Unindo as bases _training.csv_ e _test.csv_ disponíveis [aqui](https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits))

### Descrição

Classificador da base Optical Digits usando **SVM** com estratégia **1 vs ALL** usando estratégia de _cross validation_ **Holdout** com **50** iterações.

A resposta do script principal tem foco na taxa de acerto em cada iteração e média, além de mostrar a tabela de confusão média das iterações.

### Melhores Resultados
* Taxa de acertos
![](https://raw.githubusercontent.com/abnersn/svm_optical_digits/master/Imagens/linear_accuracy.png)
* Tabela de confusão
![](https://raw.githubusercontent.com/abnersn/svm_optical_digits/master/Imagens/linear_confusion.png)

### Apresentação em slides
A apresentação em slides deste trabalho pode ser acessada pelo [link](https://docs.google.com/presentation/d/1kJDI6DaE-6iJ48yJFT4ZbIYKaY7nXtM-_rr7A5dn_F0/edit?usp=sharing).
