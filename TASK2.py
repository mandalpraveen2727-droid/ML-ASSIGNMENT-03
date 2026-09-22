import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

data = {
    "Age": [22, 25, None, 30, 35, 40, None, 28],
    "Salary": [25000, 32000, 40000, None,55000, 65000, 45000, None],
    "Department": ["IT", "HR", "Finance", "IT",None, "Finance", "HR", "IT"],
    "Years_of_Experience": [1, 2, 4, None, 8, 12, 5, None]}
df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

numerical_features = ["Age","Salary","Years_of_Experience"]
categorical_features = ["Department"]

numerical_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")),("scaler", MinMaxScaler())])

categorical_pipeline = Pipeline(
    steps=[("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[("numerical",numerical_pipeline,numerical_features),
        ("categorical",categorical_pipeline,categorical_features)])

processed_data = preprocessor.fit_transform(df)

column_names = preprocessor.get_feature_names_out()

processed_df = pd.DataFrame(
    processed_data,
    columns=column_names
)

print("\nDataset After MinMaxScaler:")
print(processed_df)

print("\nMinimum Values:")
print(processed_df.min())

print("\nMaximum Values:")
print(processed_df.max())
