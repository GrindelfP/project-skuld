#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace NNI {

// --- Модуль математики ---
class Math {
public:
    static constexpr double PI = 3.14159265358979323846;

    // Умная реализация Li_n(z) для z < 0
    static double polylog(int n, double z) {
        if (std::abs(z) < 1e-15) return z;

        // Если z < -1, используем формулы инверсии (отражения)
        // Это критично для NNI, так как аргумент сигмоиды может быть большим
        if (z < -1.0) {
            double ln_z = std::log(-z);
            if (n == 1) return -std::log(1.0 - z);
            if (n == 2) return -0.5 * ln_z * ln_z - (PI * PI / 6.0) - polylog(2, 1.0 / z);
            if (n == 3) return -(1.0/6.0) * std::pow(ln_z, 3) - (PI * PI / 6.0) * ln_z - polylog(3, 1.0 / z);
            
            throw std::runtime_error("Polylog order n > 3 not implemented in this snippet.");
        }

        // Ряд для |z| <= 1
        double sum = 0.0;
        double term = z;
        for (int k = 1; k <= 120; ++k) {
            double next_term = term / std::pow(k, n);
            sum += next_term;
            term *= z;
            if (std::abs(next_term) < 1e-16) break;
        }
        return sum;
    }

    // Первообразная сигмоиды n-го порядка: F_n(u) = -Li_n(-e^u)
    static double sigmoid_integral_n(int n, double u) {
        return -polylog(n, -std::exp(u));
    }
};

// --- Класс нейросети (MLP) ---
class MLP {
private:
    int input_dim;
    int hidden_dim;
    
    // Параметры
    std::vector<std::vector<double>> weights_in_hidden;
    std::vector<double> weights_hidden_out;
    std::vector<double> bias_hidden;
    double bias_out;

public:
    MLP(int in, int hidden) : input_dim(in), hidden_dim(hidden) {
        // Простая инициализация (в реальном коде нужен Xavier/He)
        weights_in_hidden.assign(hidden, std::vector<double>(in, 0.1));
        weights_hidden_out.assign(hidden, 0.1);
        bias_hidden.assign(hidden, 0.0);
        bias_out = 0.0;
    }

    // Сеттеры для загрузки весов из вашей обученной модели C
    void setWeights(const std::vector<std::vector<double>>& w_in, 
                    const std::vector<double>& w_out, 
                    const std::vector<double>& b_h, double b_o) {
        weights_in_hidden = w_in;
        weights_hidden_out = w_out;
        bias_hidden = b_h;
        bias_out = b_o;
    }

    double forward(const std::vector<double>& x) const {
        double sum_out = bias_out;
        for (int h = 0; h < hidden_dim; ++h) {
            double activation = bias_hidden[h];
            for (int i = 0; i < input_dim; ++i) {
                activation += x[i] * weights_in_hidden[h][i];
            }
            sum_out += weights_hidden_out[h] * (1.0 / (1.0 + std::exp(-activation)));
        }
        return sum_out;
    }

    // ГЛАВНОЕ: Аналитическое интегрирование NNI
    double calculateIntegral(const std::vector<double>& mins, const std::vector<double>& maxs) const {
        double volume = 1.0;
        for (size_t i = 0; i < mins.size(); ++i) volume *= (maxs[i] - mins[i]);

        // 1. Вклад смещения выходного слоя
        double total_integral = bias_out * volume;

        // 2. Вклад скрытых нейронов
        for (int h = 0; h < hidden_dim; ++h) {
            double weight_prod = 1.0;
            for (int i = 0; i < input_dim; ++i) weight_prod *= weights_in_hidden[h][i];

            if (std::abs(weight_prod) < 1e-9) continue; // Упрощенная защита от нуля

            int num_vertices = 1 << input_dim;
            double vertex_sum = 0.0;

            for (int v = 0; v < num_vertices; ++v) {
                double linear_comb = bias_hidden[h];
                int negative_signs = 0;

                for (int i = 0; i < input_dim; ++i) {
                    if ((v >> i) & 1) {
                        linear_comb += weights_in_hidden[h][i] * maxs[i];
                    } else {
                        linear_comb += weights_in_hidden[h][i] * mins[i];
                        negative_signs++;
                    }
                }

                double val = Math::sigmoid_integral_n(input_dim, linear_comb);
                if (negative_signs % 2 == 1) vertex_sum -= val;
                else vertex_sum += val;
            }

            total_integral += weights_hidden_out[h] * (vertex_sum / weight_prod);
        }

        return total_integral;
    }
};

} // namespace NNI
