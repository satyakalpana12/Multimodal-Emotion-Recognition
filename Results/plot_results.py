import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Results/results_summary.csv")

plt.figure(figsize=(6, 4))
plt.bar(df["Model"], df["Accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("Results/model_accuracy_comparison.png")
plt.show()