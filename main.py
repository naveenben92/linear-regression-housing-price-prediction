
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

kagglehub.login()

# Download the dataset
dataset_path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

df = pd.read_csv("/kaggle/input/housing-prices-dataset/Housing.csv")

df.head()

df.fillna(df.mean(numeric_only=True), inplace=True)



num_houses = len(df)
print(f"There are {num_houses} houses in the dataset.")

# Define categorical and numeric features
# Simplified numeric features
numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Categorical features
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']

df = df[numeric_features + categorical_features + ['price']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first', dtype=float), categorical_features)
    ]
)

X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

\# ----------------------------
print("\nEnter house details to predict price:")

user_input = {}

# Numeric inputs
for feature in numeric_features:
    while True:
        try:
            user_input[feature] = float(input(f"{feature}: "))
            break
        except ValueError:
            print("Please enter a valid number.")

# Categorical inputs
for feature in categorical_features:
    while True:
        value = input(f"{feature} (yes/no): ").strip().lower()
        if value in ['yes', 'no']:
            user_input[feature] = value
            break
        else:
            print("Please enter 'yes' or 'no'.")

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Ensure column order matches training data
input_df = input_df[X.columns]

# Predict house price
predicted_price = model.predict(input_df)[0]

print(f"\nPredicted House Price: ${predicted_price:,.2f}")
