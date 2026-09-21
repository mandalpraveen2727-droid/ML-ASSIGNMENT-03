import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create the numerical dataset
data = {
    "Age": [22, 25, None, 30, 35, 40, None, 28],
    "Salary": [25000, 32000, 40000, None,55000, 65000, 45000, None],
    "Years_of_Experience": [1, 2, 4, None, 8, 12, 5, None]}

df = pd.DataFrame(data)

print("Original Numerical Dataset:")
print(df)

# Pipeline using StandardScaler
standard_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])

# Pipeline using MinMaxScaler
minmax_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", MinMaxScaler())])

# Apply both transformations
standard_scaled_data = standard_pipeline.fit_transform(df)
minmax_scaled_data = minmax_pipeline.fit_transform(df)

# Convert transformed arrays into DataFrames
standard_df = pd.DataFrame(standard_scaled_data,columns=df.columns)
minmax_df = pd.DataFrame(minmax_scaled_data,columns=df.columns)

# Display StandardScaler result
print("\nData After StandardScaler:")
print(standard_df)

print("\nStandardScaler Minimum Values:")
print(standard_df.min())

print("\nStandardScaler Maximum Values:")
print(standard_df.max())

print("\nStandardScaler Mean Values:")
print(standard_df.mean())

# Display MinMaxScaler result
print("\nData After MinMaxScaler:")
print(minmax_df)

print("\nMinMaxScaler Minimum Values:")
print(minmax_df.min())

print("\nMinMaxScaler Maximum Values:")
print(minmax_df.max())