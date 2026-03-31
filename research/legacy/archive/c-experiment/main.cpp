#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "nni.h"
// #include "nni_integration.h"
#include "nni_boost.hpp"

#define INPUT_SIZE 3
#define HIDDEN_SIZE 100

/* --- Main --- */

int main() {
    srand(time(NULL));

    int inputs = INPUT_SIZE;
    int hidden_nodes = HIDDEN_SIZE;
    
    printf("Initializing Adam-Optimized MLP: %d Inputs, %d Hidden, 1 Output\n", inputs, hidden_nodes);
    MLP *nn = create_mlp(inputs, hidden_nodes);

    // Generate Data
    double *train_x = malloc(TRAIN_SIZE * inputs * sizeof(double));
    double *train_y = malloc(TRAIN_SIZE * sizeof(double));
    double *test_x = malloc(TEST_SIZE * inputs * sizeof(double));
    double *test_y = malloc(TEST_SIZE * sizeof(double));

    for (int i = 0; i < TRAIN_SIZE; i++) {
        double x = rand_double(-1.5, 1.5); // Wider range to test approximation better
        double y = rand_double(-1.5, 1.5);
        double z = rand_double(-1.5, 1.5);
        train_x[i*3] = x; train_x[i*3+1] = y; train_x[i*3+2] = z;
        train_y[i] = target_function(x, y, z);
    }

    for (int i = 0; i < TEST_SIZE; i++) {
        double x = rand_double(-1.5, 1.5);
        double y = rand_double(-1.5, 1.5);
        double z = rand_double(-1.5, 1.5);
        test_x[i*3] = x; test_x[i*3+1] = y; test_x[i*3+2] = z;
        test_y[i] = target_function(x, y, z);
    }

    // Training Loop
    printf("Training for %d epochs using Adam...\n", EPOCHS);
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < TRAIN_SIZE; i++) {
            train(nn, &train_x[i * inputs], train_y[i]);
        }
        
        // Monitoring
        if (epoch % (EPOCHS/10) == 0) {
            double dummy[] = {0.0, 0.0, 0.0};
            double p = forward(nn, dummy);
            double t = target_function(0.0, 0.0, 0.0); // cos(2pi) = 1.0
            printf("Epoch %d | Loss (approx): %.5f | Target(0,0,0)=1.0, Pred=%.5f\n", 
                   epoch, fabs(t - p), p);
        }
    }

    // Testing
    // 4. Testing & Metrics
    printf("\n--- Final Testing (Accuracy by Decimal Places) ---\n");
    
    // Массив для хранения количества успешных попаданий для каждой позиции (от 1 до 10)
    int correct_counts[10] = {0};
    double total_error = 0;

    for (int i = 0; i < TEST_SIZE; i++) {
        double pred = forward(nn, &test_x[i * inputs]);
        double act = test_y[i];
        double err = fabs(pred - act);
        total_error += err;

        // Проверяем точность от 10^-1 до 10^-10
        for (int d = 1; d <= 10; d++) {
            double tolerance = pow(10, -d);
            if (err < tolerance) {
                correct_counts[d-1]++;
            }
        }
    }

    printf("Mean Absolute Error: %.12f\n\n", total_error / TEST_SIZE);
    printf("Precision | Hits | Percentage\n");
    printf("------------------------------\n");
    for (int d = 1; d <= 10; d++) {
        printf("%2d digits | %4d | %6.2f%%\n", 
               d, 
               correct_counts[d-1], 
               (100.0 * correct_counts[d-1] / TEST_SIZE));
    }




    // 2. Настройка границ интегрирования
    std::vector<double> mins = {-1.0, -1.0, -1.0};
    std::vector<double> maxs = {1.0, 1.0, 1.0};

    std::cout << "\n--- NNI High-Precision Integration (Boost) ---" << std::endl;

    // 3. Вызов функции интегрирования
    auto start = std::chrono::high_resolution_clock::now();
    
    double result = NNI::calculate_integral_boost(nn, mins, maxs);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 4. Вывод результата
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Analytical Integral: " << result << std::endl;
    std::cout << "Computation Time:    " << elapsed.count() << " seconds" << std::endl;



    // Cleanup
    free_mlp(nn);
    free(train_x); free(train_y); free(test_x); free(test_y);
    return 0;
}
