import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

model    = pickle.load(open('models/best_model.pkl', 'rb'))
features = ['Irradiance', 'Inlet Temp', 'Ambient Temp', 'Wind Speed', 'Tilt Angle']

# Get importance values
importance = model.feature_importances_
indices    = np.argsort(importance)

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
colors  = ['#3B8BD4' if i == indices[-1] else '#9FE1CB' for i in indices]

ax.barh(
    [features[i] for i in indices],
    importance[indices],
    color=colors, edgecolor='none', height=0.55
)

for i, (idx, val) in enumerate(zip(indices, importance[indices])):
    ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=11)

ax.set_xlabel('Importance Score', fontsize=11)
ax.set_title('Feature Importance — Solar Efficiency Prediction', fontsize=12)
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150)
plt.show()
print("Saved to outputs/feature_importance.png")