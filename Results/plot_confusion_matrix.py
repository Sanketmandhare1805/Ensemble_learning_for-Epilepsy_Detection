import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Final Ensemble Confusion Matrix
cm = [[1821, 19],
      [32, 428]]

# Labels
labels = ['Non-Seizure', 'Seizure']

# Plot
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Ensemble Confusion Matrix")

# Save image
plt.savefig("results/confusion_matrix.png")

plt.show()