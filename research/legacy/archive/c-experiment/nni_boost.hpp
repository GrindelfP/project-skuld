#ifndef NNI_BOOST_HPP
#define NNI_BOOST_HPP

#include <boost/math/special_functions/polylog.hpp>
#include <vector>
#include <cmath>
#include <iostream>

// Подключаем ваш C-заголовок. 
// extern "C" гарантирует, что C++ поймет имена функций из nni.h
extern "C" {
    #include "nni.h"
}

namespace NNI {

/**
 * Вычисляет определенный интеграл нейросети на гиперпрямоугольнике.
 * Использует boost::math::polylog для обеспечения максимальной точности.
 */
double calculate_integral_boost(MLP* net, const std::vector<double>& mins, const std::vector<double>& maxs) {
    int n = net->input_size;
    
    // 1. Интеграл смещения выходного слоя (Bias): b_out * Volume
    double volume = 1.0;
    for (int i = 0; i < n; ++i) volume *= (maxs[i] - mins[i]);
    double integral = net->b_output * volume;

    // 2. Интеграл каждого скрытого нейрона
    for (int h = 0; h < net->hidden_size; ++h) {
        // Произведение весов в знаменателе (w1 * w2 * ... * wn)
        double weight_prod = 1.0;
        for (int i = 0; i < n; ++i) weight_prod *= net->w_input_hidden[i][h];

        // В статье указано, что если веса близки к 0, нужны альтернативные формулы.
        // Здесь используем простую проверку для стабильности.
        if (std::abs(weight_prod) < 1e-12) continue;

        // Перебор всех 2^n вершин гиперпрямоугольника
        int num_vertices = 1 << n;
        double vertex_sum = 0.0;

        for (int v = 0; v < num_vertices; ++v) {
            double u = net->b_hidden[h]; // Линейная комбинация: b + sum(w_i * x_i)
            int min_count = 0;

            for (int i = 0; i < n; ++i) {
                if ((v >> i) & 1) {
                    u += net->w_input_hidden[i][h] * maxs[i];
                } else {
                    u += net->w_input_hidden[i][h] * mins[i];
                    min_count++; // Считаем количество нижних границ для определения знака
                }
            }

            // Вычисляем первообразную n-го порядка: F_n(u) = -Li_n(-e^u)
            // Используем Boost для вычисления полилогарифма порядка n
            // Для вашей задачи n = 3
            double fn_u = -boost::math::polylog(n, -std::exp(u));

            // Знак определяется четностью количества нижних границ (формула Ньютона-Лейбница для n-мерного случая)
            if (min_count % 2 == 1) vertex_sum -= fn_u;
            else vertex_sum += fn_u;
        }

        // Вклад нейрона: w_out * (vertex_sum / product_of_weights)
        integral += net->w_hidden_output[h] * (vertex_sum / weight_prod);
    }

    return integral;
}

} // namespace NNI

#endif