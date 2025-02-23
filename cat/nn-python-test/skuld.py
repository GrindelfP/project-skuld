import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

import numpy as np
from mpmath import polylog

from sklearn.model_selection import train_test_split


class MLP(nn.Module):
    """
        Нейросеть, которая будет обучаться приближать функцию одной переменной.

        Нейросеть имеет архитектуру:

        Входной слой (1 нейрон для переменной функции + смещение, линейная функция активации)
        Скрытый слой (произвольное количество нейронов + смещение, функция активации - сигмоида)
        Выходной слой (1 нейрон для приближенного значения функции, линейная функция активации)
    """

    def __init__(self, input_size, hidden_size):
        """
            Конструктор для нейросети.
            @param self        нейросеть (необходим для включения в класс)
            @param hidden_size размер скрытого слоя (размеры входного и выходного слоёв равны одному
                               в рамках данной задачи, так как у функции одна переменная и
                               задача сводится к описанию функции, то есть числа, скаляра).
        """
        super(MLP, self).__init__()
        self.input_hidden_layer = nn.Linear(input_size, hidden_size)  # инициализация входного и скрытого слоя,
        # размеры: 1 --> размер скрытого слоя
        self.sigmoid_activation = nn.Sigmoid()  # инициализация функции активации скрытого слоя
        self.output_layer = nn.Linear(hidden_size, 1)  # инициализация выходного слоя,
        # размеры: размер скрытого слоя --> 1

    def forward(self, x):
        """
            Функция распространения данных через нейросеть вперёд.

            @param self   нейросеть (необходим для включения в класс)
            @param x      данные

            @returns выход в выходном нейроне
        """
        x = self.input_hidden_layer(x)  # данные прошли входной слой и аккумулирвоаны в скрытом слое
        x = self.sigmoid_activation(x)  # данные прошли функцию активации скрытого слоя
        x = self.output_layer(x)  # данные прошли выходной слой

        return x


def train_model(model, criterion, optimizer, x_train, y_train, epochs):
    """
        Trains the model.

        @param model        The model to be trained
        @param criterion    Loss function
        @param optimizer    Optimization algorithm
        @param x_train      Training inputs
        @param y_train      True labels
        @param epochs       Number of training epochs
    """
    loss_history = []  # история обучения (изменения функции потерь)
    for epoch in range(epochs):
        predictions = model(x_train)  # все переменные проводятся через нейросеть
        # и формируются предскзания значений функции
        loss = criterion(predictions, y_train)  # вычисляется функция потерь на данной эпохе

        optimizer.zero_grad()  # обнуляются градиенты перед обратным распространением ошибки
        loss.backward()  # обратное распространение ошибки
        optimizer.step()  # шаг оптимизации - обновление параметров модели

        loss_history.append(loss.item())  # запись текущей функции потерь

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.10f}')  # вывод информации об обучении

    return loss_history  # возвращается история обучения


def test_model(model, criterion, x_test, y_test):
    """
        Tests the model.

        @param model        The trained model
        @param criterion    Loss function
        @param x_test       Test inputs
        @param y_test       True labels
    """
    with torch.no_grad():  # отключение расчета градиентов
        # (расчет градиентов может происходить по умолчанию
        # даже без использования их потом, что излишне нагружает память)
        predictions = model(x_test)  # тестовые переменные проводятся через обученную модель
        loss = criterion(predictions, y_test)  # вычисляется функция потерь для тестового набора

    # Возвращаем вычисленную функцию потерь
    return loss.item()  # Возвращаем скалярное значение ошибки


def predict_with_model(model, x_test):
    """
        Uses the model to predict values based on x_test arguments.

        @param model        The trained model
        @param x_test       Test inputs
    """
    with torch.no_grad():
        predictions = model(x_test)

    return predictions


def extract_model_params(model):
    """
        Функция извлечения параметров нейросети.
        @param model модель, из которой необходимо извлеч параметры
        @returns 4 объекта типа numpy.array: смещения 1-го слоя, веса 1-го слоя,
                 смещения 2-го слоя, веса 2-го слоя
    """
    # detach() - возвращает выбранный параметр, numpy() конвертирует в формат numpy.array,
    # flatten() для весов преобразует векторы-столбцы в векторы-строки.
    b1 = model.input_hidden_layer.bias.detach().numpy()
    w1 = model.input_hidden_layer.weight.detach().numpy()
    b2 = model.output_layer.bias.detach().numpy()
    w2 = model.output_layer.weight.detach().numpy().flatten()

    return b1, w1, b2, w2


def get_NN_integral(alpha1, beta1, alpha2, beta2, b1, w1, b2, w2):
    """
        Функция, реализующая метод численного интегрирования функции одной переменной
        на основе параметров нейросети. Реализует формулы (6.1) и (6.2).

        @param alpha нижняя граница интегрирования
        @param beta  верхняя граница интегрирования
        @param b1    смещения между входным и скрытым слоями
        @param w1    веса между входным и скрытым слоями
        @param b2    смещения между скрытым и выходным слоями
        @param w2    веса между скрытым и выходным слоями

        @returns численный интеграл на основе параметров нейросети.
    """

    def Phi_j(alpha1, beta1, alpha2, beta2, b1_j, w1_1j, w1_2j):
        """
            Вложенная функция, реализующая разность полилогарифмов (6.2).

            @param alpha нижняя граница интегрирования
            @param beta  верхняя граница интегрирования
            @param b1_j  j-е смещение между входным и скрытым слоями
            @param w1_j  j-тый вес между входным и скрытым слоями

            @returns разность полилогарифмов (6.2)
        """
        term_1 = polylog(2, -np.exp(-b1_j - w1_1j * alpha1 - w1_2j * alpha2))
        term_2 = polylog(2, -np.exp(-b1_j - w1_1j * alpha1 - w1_2j * beta2))
        term_3 = polylog(2, -np.exp(-b1_j - w1_1j * beta1 - w1_2j * alpha2))
        term_4 = polylog(2, -np.exp(-b1_j - w1_1j * beta1 - w1_2j * beta2))

        return term_1 - term_2 - term_3 + term_4

    bounds_factor = (beta2 - alpha2) * (beta1 - alpha1)

    integral_sum = 0

    for w2_j, w1_1j, w1_2j, b1_j in zip(w2, w1[0], w1[1], b1):
        phi_j = Phi_j(alpha1, beta1, alpha2, beta2, b1_j, w1_1j, w1_2j)
        integral_sum += w2_j * (bounds_factor + phi_j / w1_1j * w1_2j)

    return b2 * integral_sum + integral_sum


def integrate(alpha1, beta1, alpha2, beta2, model):
    b1, w1, b2, w2 = extract_model_params(model)
    return get_NN_integral(alpha1, beta1, alpha2, beta2, b1, w1, b2, w2)