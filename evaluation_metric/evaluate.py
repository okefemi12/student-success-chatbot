import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ai_generated_questions_labeled.csv")
distribution = data["bloom_level"].value_counts(normalize=True) * 100
distribution = distribution.sort_index()

print("=== Bloom’s Taxonomy Distribution (%) ===")
print(distribution)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.bar(distribution.index, distribution.values, color="skyblue", edgecolor="black")
plt.title("Distribution of Chatbot Responses by Bloom’s Taxonomy", fontsize=14)
plt.xlabel("Cognitive Level", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
