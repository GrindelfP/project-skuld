import matplotlib.pyplot as plt
a = [10, 1, 0.3, 0.2, 0.1, 12, 199]
plt.semilogy(a, 'b-', linewidth=1.5, marker='o', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('MAE Loss (log scale)')
plt.title('Training MAE Loss Over Epochs')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()
plt.show()
