#ifndef NNI_INTEGRATION_H
#define NNI_INTEGRATION_H

#include "nni.h"

// Приблизительное вычисление полилогарифма Li_n(z) через ряд для z < 0
// В статье используется Li_n(-e^u). 
// Для n=1: Li_1(z) = -ln(1-z)
// Для n=2,3: используем степенной ряд (достаточно для демонстрации)
double polylog(int n, double z) {
    if (n == 1) return -log(1.0 - z);
    
    double sum = 0.0;
    double term = z;
    for (int k = 1; k <= 100; k++) {
        sum += term / pow(k, n);
        term *= z;
        if (fabs(term) < 1e-12) break;
    }
    return sum;
}

// Первообразная n-го порядка для сигмоиды sigma(u)
// F_n(u) = -Li_n(-e^u)
double sigmoid_integral_n(int n, double u) {
    return -polylog(n, -exp(u));
}

// Вычисление интеграла NNI для MLP в n-мерном пространстве
// mins и maxs - массивы границ интегрирования для каждой оси
double nni_calculate_integral(MLP *net, double *mins, double *maxs) {
    int n = net->input_size;
    double total_integral = 0.0;

    // 1. Интеграл константы (bias выходного слоя)
    double volume = 1.0;
    for (int i = 0; i < n; i++) volume *= (maxs[i] - mins[i]);
    total_integral += net->b_output * volume;

    // 2. Интеграл суммы нейронов
    for (int h = 0; h < net->hidden_size; h++) {
        double neuron_contribution = 0.0;
        
        // В 3D случае это сумма по 2^3 = 8 вершинам параллелепипеда
        // Мы используем формулу Ньютона-Лейбница n раз.
        // Для 3D: I = (1 / (w1*w2*w3)) * sum( (-1)^k * F_3(W*v + b) )
        
        double weight_prod = 1.0;
        for (int i = 0; i < n; i++) weight_prod *= net->w_input_hidden[i][h];

        // Перебор всех 2^n вершин
        int num_vertices = 1 << n;
        double vertex_sum = 0.0;
        
        for (int v = 0; v < num_vertices; v++) {
            double linear_comb = net->b_hidden[h];
            int sign = 0;
            
            for (int i = 0; i < n; i++) {
                if ((v >> i) & 1) {
                    linear_comb += net->w_input_hidden[i][h] * maxs[i];
                } else {
                    linear_comb += net->w_input_hidden[i][h] * mins[i];
                    sign++;
                }
            }
            
            double val = sigmoid_integral_n(n, linear_comb);
            if (sign % 2 == 1) vertex_sum -= val;
            else vertex_sum += val;
        }

        neuron_contribution = net->w_hidden_output[h] * (vertex_sum / weight_prod);
        total_integral += neuron_contribution;
    }

    return total_integral;
}

#endif
