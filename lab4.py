import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_percentage_error,
    r2_score
)

print("\n--- Q1: Audio Classification ---")

csv_path = r"C:\Users\prana\OneDrive\Desktop\Viva Rate\audio_labels_20250802_145147.csv"
audio_dir = r"C:\Users\prana\OneDrive\Desktop\Viva Rate\Audio_dataset\Cleaned_Audios"

ds = pd.read_csv(csv_path)
print("CSV Columns:", ds.columns.tolist())

file_column = 'filename' if 'filename' in ds.columns else 'file_path'

features = []
valid_rows = []

for idx, row in ds.iterrows():
    file_name = row[file_column]
    full_path = os.path.join(audio_dir, file_name)
    
    try:
        y, sr = librosa.load(full_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.append(np.mean(mfcc, axis=1))
        valid_rows.append(row)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

if not features:
    raise RuntimeError("No audio files could be processed. .")

valid_df = pd.DataFrame(valid_rows)
valid_df['features'] = features

X_audio = np.vstack(valid_df['features'].values)
y_audio = valid_df['label'].map({'clear': 1, 'unclear': 0})

X_train_audio, X_test_audio, y_train_audio, y_test_audio = train_test_split(
    X_audio, y_audio, test_size=0.2, random_state=42)

knn_audio = KNeighborsClassifier(n_neighbors=3)
knn_audio.fit(X_train_audio, y_train_audio)

y_pred_train_audio = knn_audio.predict(X_train_audio)
y_pred_test_audio = knn_audio.predict(X_test_audio)

print("Test Confusion Matrix:\n", confusion_matrix(y_test_audio, y_pred_test_audio))
print(classification_report(y_test_audio, y_pred_test_audio))
print("Train Confusion Matrix:\n", confusion_matrix(y_train_audio, y_pred_train_audio))
print(classification_report(y_train_audio, y_pred_train_audio))

print("\n--- Q2: Stock Price Regression ---")

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
df['Day_num'] = df['Day'].astype('category').cat.codes
df['Month_num'] = df['Month'].astype('category').cat.codes

X_stock = df[['Day_num', 'Month_num']]
y_stock = df['Price']

X_train_stock, X_test_stock, y_train_stock, y_test_stock = train_test_split(
    X_stock, y_stock, test_size=0.2, random_state=42)

regressor = KNeighborsRegressor(n_neighbors=3)
regressor.fit(X_train_stock, y_train_stock)
y_pred_stock = regressor.predict(X_test_stock)

mse = mean_squared_error(y_test_stock, y_pred_stock)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_stock, y_pred_stock) * 100
r2 = r2_score(y_test_stock, y_pred_stock)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.2f}")

print("\n--- Q3: Visualizing Random Points ---")

X_vis = np.random.uniform(1, 10, size=(20, 2))
y_vis = np.random.choice([0, 1], size=20)

for i in range(20):
    plt.scatter(X_vis[i, 0], X_vis[i, 1], color='blue' if y_vis[i] == 0 else 'red')
plt.title("Q3: Random Points Classification")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.show()

print("\n--- Q4: Grid Classification ---")

x_vals = np.arange(0, 10.1, 0.1)
y_vals = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_vals, y_vals)
X_grid = np.c_[xx.ravel(), yy.ravel()]

model_k3 = KNeighborsClassifier(n_neighbors=3)
model_k3.fit(X_vis, y_vis)
y_pred_grid = model_k3.predict(X_grid)

plt.figure(figsize=(8, 6))
plt.scatter(X_grid[:, 0], X_grid[:, 1], c=y_pred_grid, cmap='bwr', alpha=0.3, s=5)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap='bwr', edgecolors='black', s=50)
plt.title("Q4: k=3 Classification of Grid")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

print("\n--- Q5: Grid Classification for Multiple k ---")

k_values = [1, 5, 7, 9]
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_vis, y_vis)
    y_pred_k = model.predict(X_grid)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_grid[:, 0], X_grid[:, 1], c=y_pred_k, cmap='bwr', alpha=0.3, s=5)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap='bwr', edgecolors='black', s=50)
    plt.title(f"Q5: k={k}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

print("\n--- Q7: Hyperparameter Tuning using GridSearchCV ---")

param_grid = {'n_neighbors': list(range(1, 11))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=5)
grid.fit(X_vis, y_vis)

print("Best k:", grid.best_params_['n_neighbors'])
print("Best cross-val accuracy:", grid.best_score_)