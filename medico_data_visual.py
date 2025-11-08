import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('medical_examination.csv')
print(df.head())
# Step 1: Add BMI column
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

# Step 2: Create an 'overweight' column (1 = overweight, 0 = not)
df['overweight'] = (df['BMI'] > 25).astype(int)

print(df[['height', 'weight', 'BMI', 'overweight']].head())
# Normalize cholesterol and glucose values
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

print(df[['cholesterol', 'gluc']].head())

# Step 3: Prepare data for categorical plot
df_cat = pd.melt(df, id_vars=['cardio'], 
                 value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

# Group and calculate the mean for each feature per cardio value
df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()

# Step 4: Draw the categorical plot
sns.catplot(x='variable', y='size', hue='value', col='cardio', data=df_cat, kind='bar')

plt.show()
# ================================
# PART 2: Correlation Heatmap
# ================================

import numpy as np

# Step 1: Clean the data
df_heat = df[(df['ap_hi'] >= df['ap_lo']) &
             (df['height'] >= df['height'].quantile(0.025)) &
             (df['height'] <= df['height'].quantile(0.975)) &
             (df['weight'] >= df['weight'].quantile(0.025)) &
             (df['weight'] <= df['weight'].quantile(0.975))]

# Step 2: Calculate correlation matrix
corr = df_heat.corr()

# Step 3: Generate mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Step 4: Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.title("Correlation Heatmap of Medical Data", fontsize=14)
plt.show()
plt.show(block=True)



