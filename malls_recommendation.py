import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


df = pd.read_csv("malls.csv")
print("First 5 rows of dataset:\n", df.head())

features = ["Num_Shops", "Num_Brands", "Has_FoodCourt", "Has_Cinema",
            "Avg_Rating", "Luxury_Shops", "Footfall"]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

cmapping = {0: "Premium", 1: "Budget", 2: "Standard"}
df["Category"] = df["Cluster"].map(cmapping)

print("\nClustered Data with Categories:\n", df[["MallName", "Cluster", "Category"]])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, df["Cluster"], test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # hl 1
    Dense(32, activation='relu'),  # hl 2
    Dense(3, activation='softmax')  # it is the output layer
])

model.compile(optimizer=Adam(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=2,
                    validation_data=(X_test, y_test), verbose=0)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")

prediction = model.predict(X_test) # it gives probablity
pred = np.argmax(prediction, axis=1) # it gives output with maximum probablity 

unique_classes = np.unique(y_test)
target_names = [cmapping[c] for c in unique_classes]

cm = confusion_matrix(y_test, pred, labels=unique_classes)
print("The confusion Matrix is :\n", cm)

print("\nClassification Report:\n", classification_report(y_test, pred,labels=unique_classes,target_names=target_names))

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.show()

def recommend_malls(category):
    malls = df[df["Category"] == category][["MallName", "Avg_Rating", "Footfall"]]
    if malls.empty:
        return f"No malls found for category '{category}'"
    return malls.sort_values(by=["Avg_Rating", "Footfall"], ascending=False)

user_choice = "Premium"
print(f"\nRecommended {user_choice} malls:\n", recommend_malls(user_choice))