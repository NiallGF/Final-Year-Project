# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import pandas as pd
import os

# Directory containing NHANES datasets
data_dir = '/workspaces/Final-Year-Project/Data For Final Proj/NHANES Actual/2017-2018/'

# NHANES variables of interest
nhanes_files_variables = {
    'P_GHB.xpt': ['SEQN', 'LBXGH'],  # Glycohemoglobin
    'P_GLU.xpt': ['SEQN', 'LBXGLU'],  # Plasma Fasting Glucose
    'P_ALQ.xpt': ['SEQN', 'ALQ130'],  # Alcohol Use
    'P_BPQ.xpt': ['SEQN', 'BPQ020', 'BPQ080'],  # Blood Pressure & Cholesterol
    'P_CDQ.xpt': ['SEQN', 'CDQ001', 'CDQ010'],  # Cardiovascular Health
    'P_CBQPFA.xpt': ['SEQN', 'CBQ506', 'CBQ860'],  # Consumer Behavior
    'P_DEQ.xpt': ['SEQN', 'DED120', 'DED125'],  # Dermatology
    'P_DBQ.xpt': ['SEQN', 'DBQ700', 'DBQ197'],  # Diet Behavior
    'P_FSQ.xpt': ['SEQN', 'FSDHH'],  # Food Security
    'P_HIQ.xpt': ['SEQN', 'HIQ011'],  # Health Insurance
    'P_HEQ.xpt': ['SEQN', 'HEQ010', 'HEQ030'],  # Hepatitis
    'P_HUQ.xpt': ['SEQN', 'HUQ010'],  # Hospital Utilization
    'P_INQ.xpt': ['SEQN', 'INDFMMPI'],  # Income
    'P_MCQ.xpt': ['SEQN', 'MCQ010', 'MCQ160A'],  # Medical Conditions
    'P_DPQ.xpt': ['SEQN', 'DPQ020'],  # Mental Health
    'P_PAQ.xpt': ['SEQN', 'PAQ650', 'PAQ665'],  # Physical Activity
    'P_SMQ.xpt': ['SEQN', 'SMQ020', 'SMQ040'],  # Smoking
    'P_WHQ.xpt': ['SEQN', 'WHD020', 'WHD050'],  # Weight History
    'P_DEMO.xpt': ['SEQN', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR']  # Demographics
}

# Initialize an empty DataFrame for the final merged dataset
df = None

# Loop through and load datasets
for file_name, variables in nhanes_files_variables.items():
    file_path = os.path.join(data_dir, file_name)
    try:
        temp_df = pd.read_sas(file_path, format="xport")[variables]
        print(f"Loaded {file_name} with {temp_df.shape[0]} rows and {temp_df.shape[1]} columns.")
        # Merge datasets on SEQN using outer join
        if df is None:
            df = temp_df
        else:
            df = pd.merge(df, temp_df, on="SEQN", how="outer")
    except Exception as e:
        print(f"Error loading {file_name}: {e}")

# Display the first few rows and the DataFrame info
print("Merged DataFrame (first few rows):")
print(df.head())

# Display the DataFrame info
df.info()


# %%
# Define thresholds for diabetes
hba1c_threshold = 6.5  # HbA1c percentage
glucose_threshold = 126  # Fasting glucose in mg/dL

# Ensure relevant columns exist in the DataFrame
critical_columns = ['LBXGH', 'LBXGLU']  # HbA1c, Glucose

# Check if critical columns are present
if all(col in df.columns for col in critical_columns):
    # Create the `diabetic` variable
    df['diabetic'] = (
        (df['LBXGH'] >= hba1c_threshold) |  # HbA1c >= 6.5%
        (df['LBXGLU'] >= glucose_threshold)  # Fasting glucose >= 126 mg/dl
    ).astype(int)  # Ensure the resulting column is binary (0 or 1)

    # Print value counts for the new `diabetic` variable
    print("Diabetic variable value counts:")
    print(df['diabetic'].value_counts())
else:
    print(f"One or more critical columns missing: {critical_columns}")




# %% [markdown]
# 

# %%
# Drop the original columns
columns_to_drop = ['LBXGH', 'LBXGLU', 'DIQ010']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Confirm the updated DataFrame
print("Updated DataFrame shape:", df.shape)
print("Columns in the updated DataFrame:", df.columns)



# %%
# Drop columns where the number of non-null values is less than 8000
df_cleaned = df.dropna(thresh=6000, axis=1)

# Check the result
df_cleaned = df_cleaned.dropna()
df_cleaned.info()


# %%
df_cleaned.diabetic.value_counts()



# %%
# Calculate the correlation matrix
corr_matrix = df_cleaned.corr()

# Plot the correlation matrix
plt.figure(figsize=(16, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# %%



