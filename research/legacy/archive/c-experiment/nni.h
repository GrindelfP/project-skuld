#ifndef NNI_H
#define NNI_H


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* --- Configuration --- */
#define LEARNING_RATE 0.0001  // Adam usually works best with lower rates (0.001 to 0.01)
#define EPOCHS 1000        // Fewer epochs needed for Adam to converge
#define TRAIN_SIZE 1000
#define TEST_SIZE 200

// Adam Hyperparameters
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --- Data Structures --- */

typedef struct {
    int input_size;
    int hidden_size;
    long long t; // Time step counter for Adam

    // --- Weights & Biases ---
    double **w_input_hidden;
    double *w_hidden_output;
    double *b_hidden;
    double b_output;

    // --- Adam Optimizer State (First Moment 'm') ---
    double **m_w_input_hidden;
    double *m_w_hidden_output;
    double *m_b_hidden;
    double m_b_output;

    // --- Adam Optimizer State (Second Moment 'v') ---
    double **v_w_input_hidden;
    double *v_w_hidden_output;
    double *v_b_hidden;
    double v_b_output;

    // Internal states for backprop
    double *hidden_outputs;
} MLP;

/* --- Helper Functions --- */

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

double rand_double(double min, double max) {
    return min + (rand() / (double) RAND_MAX) * (max - min);
}

// Target: f(x, y, z) = cos(2pi + x + y + z)
double target_function(double x, double y, double z) {
    return cos((2 * M_PI) + x + y + z);
}

/* --- Memory Management --- */

MLP* create_mlp(int input_size, int hidden_size) {
    MLP *net = (MLP*)malloc(sizeof(MLP));
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->t = 0;

    // Allocations
    net->w_input_hidden = (double**)malloc(input_size * sizeof(double*));
    net->m_w_input_hidden = (double**)malloc(input_size * sizeof(double*));
    net->v_w_input_hidden = (double**)malloc(input_size * sizeof(double*));
    
    for (int i = 0; i < input_size; i++) {
        net->w_input_hidden[i] = (double*)malloc(hidden_size * sizeof(double));
        net->m_w_input_hidden[i] = (double*)calloc(hidden_size, sizeof(double)); // Init to 0
        net->v_w_input_hidden[i] = (double*)calloc(hidden_size, sizeof(double)); // Init to 0
    }

    net->w_hidden_output = (double*)malloc(hidden_size * sizeof(double));
    net->m_w_hidden_output = (double*)calloc(hidden_size, sizeof(double));
    net->v_w_hidden_output = (double*)calloc(hidden_size, sizeof(double));

    net->b_hidden = (double*)malloc(hidden_size * sizeof(double));
    net->m_b_hidden = (double*)calloc(hidden_size, sizeof(double));
    net->v_b_hidden = (double*)calloc(hidden_size, sizeof(double));

    net->hidden_outputs = (double*)malloc(hidden_size * sizeof(double));

    // Initialization
    for (int i = 0; i < input_size; i++) {
        for (int h = 0; h < hidden_size; h++) {
            net->w_input_hidden[i][h] = rand_double(-0.5, 0.5);
        }
    }
    for (int h = 0; h < hidden_size; h++) {
        net->w_hidden_output[h] = rand_double(-0.5, 0.5);
        net->b_hidden[h] = 0.0;
    }
    net->b_output = 0.0;
    net->m_b_output = 0.0;
    net->v_b_output = 0.0;

    return net;
}

void free_mlp(MLP *net) {
    for (int i = 0; i < net->input_size; i++) {
        free(net->w_input_hidden[i]);
        free(net->m_w_input_hidden[i]);
        free(net->v_w_input_hidden[i]);
    }
    free(net->w_input_hidden); free(net->m_w_input_hidden); free(net->v_w_input_hidden);
    
    free(net->w_hidden_output); free(net->m_w_hidden_output); free(net->v_w_hidden_output);
    free(net->b_hidden); free(net->m_b_hidden); free(net->v_b_hidden);
    free(net->hidden_outputs);
    free(net);
}

/* --- Optimizer Logic --- */

// The core Adam update calculation for a single parameter
void apply_adam(double *weight, double *m, double *v, double grad, double beta1_t, double beta2_t) {
    // 1. Update biased first moment estimate
    *m = BETA1 * (*m) + (1.0 - BETA1) * grad;
    
    // 2. Update biased second raw moment estimate
    *v = BETA2 * (*v) + (1.0 - BETA2) * (grad * grad);

    // 3. Compute bias-corrected first moment estimate
    double m_hat = *m / (1.0 - beta1_t);

    // 4. Compute bias-corrected second raw moment estimate
    double v_hat = *v / (1.0 - beta2_t);

    // 5. Update weight
    *weight += LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);
}

/* --- Core MLP Functions --- */

double forward(MLP *net, double *inputs) {
    // Input -> Hidden
    for (int h = 0; h < net->hidden_size; h++) {
        double activation = 0.0;
        for (int i = 0; i < net->input_size; i++) {
            activation += inputs[i] * net->w_input_hidden[i][h];
        }
        activation += net->b_hidden[h];
        net->hidden_outputs[h] = sigmoid(activation);
    }

    // Hidden -> Output
    double output = 0.0;
    for (int h = 0; h < net->hidden_size; h++) {
        output += net->hidden_outputs[h] * net->w_hidden_output[h];
    }
    output += net->b_output;
    return output;
}

void train(MLP *net, double *inputs, double target) {
    // 1. Forward Pass
    double output = forward(net, inputs);

    // 2. Calculate Gradients
    // Error = (Target - Output).
    // Note: We use gradient ASCENT logic (adding to weights) with (Target - Output).
    double output_error = (target - output);
    double output_delta = output_error; // Linear derivative is 1

    double *hidden_errors = (double*)malloc(net->hidden_size * sizeof(double));
    for (int h = 0; h < net->hidden_size; h++) {
        hidden_errors[h] = output_delta * net->w_hidden_output[h] * sigmoid_derivative(net->hidden_outputs[h]);
    }

    // 3. Adam Updates
    net->t++; // Increment time step
    
    // Pre-calculate bias corrections for efficiency (pow is expensive)
    double beta1_t = pow(BETA1, net->t);
    double beta2_t = pow(BETA2, net->t);

    // Update Hidden -> Output Weights
    for (int h = 0; h < net->hidden_size; h++) {
        double grad = output_delta * net->hidden_outputs[h];
        apply_adam(&net->w_hidden_output[h], &net->m_w_hidden_output[h], &net->v_w_hidden_output[h], grad, beta1_t, beta2_t);
    }
    // Update Output Bias
    apply_adam(&net->b_output, &net->m_b_output, &net->v_b_output, output_delta, beta1_t, beta2_t);

    // Update Input -> Hidden Weights
    for (int i = 0; i < net->input_size; i++) {
        for (int h = 0; h < net->hidden_size; h++) {
            double grad = hidden_errors[h] * inputs[i];
            apply_adam(&net->w_input_hidden[i][h], &net->m_w_input_hidden[i][h], &net->v_w_input_hidden[i][h], grad, beta1_t, beta2_t);
        }
    }
    // Update Hidden Biases
    for (int h = 0; h < net->hidden_size; h++) {
        apply_adam(&net->b_hidden[h], &net->m_b_hidden[h], &net->v_b_hidden[h], hidden_errors[h], beta1_t, beta2_t);
    }

    free(hidden_errors);
}

#endif
