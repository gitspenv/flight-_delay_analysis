import pandas as pd
import pickle
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
print("📦 Loading dataset...")
df = pd.read_csv("data/processed/df_cleaned_ready_for_modeling.csv")

# Defining the targets and features
targets = ['Dep_Delay', 'Arr_Delay']
features = [
    'Day_Of_Week', 'Airline', 'Dep_Airport', 'DepTime_label',
    'Flight_Duration', 'Distance_type', 'Delay_Carrier', 'Delay_Weather', 'Delay_NAS',
    'Delay_Security', 'Delay_LastAircraft', 'Manufacturer', 'Aicraft_age',
    'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'pres',
    'Aircraft_Age_Group'
]

# Drop missing target rows and shuffle
df = df.dropna(subset=targets).sample(frac=1, random_state=42).reset_index(drop=True)

# splitting the dataset into 40% train, 40% val, 20% test
n = len(df)
train_df = df.iloc[:int(0.4 * n)]
val_df   = df.iloc[int(0.4 * n):int(0.8 * n)]
test_df  = df.iloc[int(0.8 * n):]

X_train, y_train = train_df[features], train_df[targets]
X_val, y_val     = val_df[features], val_df[targets]
X_test, y_test   = test_df[features], test_df[targets]

# Preprocessing pipeline
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Fitting and transforming the training data
print("⚙️ Preprocessing training data...")
X_train_processed = preprocessor.fit_transform(X_train)

# Training models with a progress bar
models = {}
print("Training models...")
for target in tqdm(targets, desc="Training progress", unit="target"):
    model = LinearRegression()
    model.fit(X_train_processed, y_train[target])
    models[target] = model

# Validating models
print("\n📈 Validation Results:")
X_val_processed = preprocessor.transform(X_val)
for target in targets:
    y_pred = models[target].predict(X_val_processed)
    print(f"\n🔍 {target} Validation:")
    print(f"MAE:  {mean_absolute_error(y_val[target], y_pred):.2f}")
    print(f"RMSE: {mean_squared_error(y_val[target], y_pred):.2f}")
    print(f"R²:   {r2_score(y_val[target], y_pred):.2f}")

# Save model and preprocessor
print("\n💾 Saving models...")
os.makedirs("model", exist_ok=True)

with open("model/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("model/flight_delay_model.pkl", "wb") as f:
    pickle.dump(models, f)

# Saving the test set for later evaluation
X_test.to_csv("model/X_test.csv", index=False)
y_test.to_csv("model/y_test.csv", index=False)

print("\n Models saved in 'model/' and test set stored for later evaluation.")
