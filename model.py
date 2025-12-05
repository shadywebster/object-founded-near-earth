import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 9))

# Base scatter plot: Color by true 'is_hazardous' status
sns.scatterplot(
    data=combined_df,
    x='V relative (km/s)',
    y='Cleaned_Diameter_km',
    hue='is_hazardous',
    palette='viridis',
    alpha=0.6,
    s=50
)

# Filter for XGBoost predicted hazardous objects
predicted_hazardous_xgb = combined_df[combined_df['xgb_predicted_hazardous'] == 1].copy()

# Overlay points predicted as hazardous by XGBoost
plt.scatter(
    predicted_hazardous_xgb['V relative (km/s)'],
    predicted_hazardous_xgb['Cleaned_Diameter_km'],
    color='red',
    marker='X',
    s=100,
    label='XGBoost Predicted Hazardous (1)',
    alpha=0.8,
    linewidth=1.5
)

# Select a small sample for annotation
sample_to_annotate = predicted_hazardous_xgb.sample(n=min(10, len(predicted_hazardous_xgb)), random_state=42)

# Add text annotations for sampled objects
for i, row in sample_to_annotate.iterrows():
    plt.annotate(
        row['Object'],
        (row['V relative (km/s)'], row['Cleaned_Diameter_km']),
        textcoords="offset points",
        xytext=(5, 5),
        ha='left',
        fontsize=8,
        color='darkgreen'
    )

plt.title('V relative (km/s) vs. Cleaned_Diameter_km by True Status and XGBoost Prediction')
plt.xlabel('V relative (km/s) (Scaled)')
plt.ylabel('Cleaned_Diameter_km (Scaled)')
plt.legend(title='Legend', loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
