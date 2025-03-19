import matplotlib.pyplot as plt


def plot_test_results(results: list):
    x_values = [i for i, _ in enumerate(results)]  # Index as x-axis
    nnis = [v1 for v1, v2, v3 in results]
    quads = [v2 for v1, v2, v3 in results]
    trapz = [v3 for v1, v2, v3 in results]
    abs_err_quad = [abs(v1 - v2) for v1, v2, v3 in results]
    abs_err_trapz = [abs(v1 - v3) for v1, v2, v3 in results]
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 5))
    
    ax1.plot(x_values, nnis, label='nnis', marker='o')
    ax1.plot(x_values, quads, label='quads', marker='x')
    ax1.plot(x_values, trapz, label='trapz', marker='s')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Integrals')
    ax1.set_title('Numerical integral values')
    ax1.legend()
    
    ax2.plot(x_values, abs_err_quad, label='|nnis - quads|', marker='s', color='red')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Absolute error')
    ax2.set_title('NN and quad method')
    ax2.legend()
    
    ax3.plot(x_values, abs_err_trapz, label='|nnis - trapz|', marker='o', color='red')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Absolute error')
    ax3.set_title('NN and trapezoidal method')
    ax3.legend()
    
    ax4.plot(x_values, nnis, label='nnis', marker='o')
    ax4.set_xlabel('Index')
    ax4.set_ylabel('NNIs')
    ax4.set_title('NN integrals')
    ax4.legend()
    
    ax5.plot(x_values, quads, label='quads', marker='x', color='y')
    ax5.set_xlabel('Index')
    ax5.set_ylabel('QUADs')
    ax5.set_title('QUAD integrals')
    ax5.legend()
    
    ax6.plot(x_values, trapz, label='trapz', marker='s', color='g')
    ax6.set_xlabel('Index')
    ax6.set_ylabel('TRAPZ')
    ax6.set_title('Trapezoid integrals')
    ax6.legend()
    
    plt.tight_layout()
    plt.show()
