import os, zipfile, cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

files.upload()  # загрузите kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d jaidalmotra/road-sign-detection
with zipfile.ZipFile("road-sign-detection.zip", "r") as zip_ref:
    zip_ref.extractall("dataset")

IMG_SIZE = 64
DATA_PATH = "dataset/road_sign_detection/train"

def load_images(path, size):
    imgs, lbls = [], []
    for label in os.listdir(path):
        for file in os.listdir(os.path.join(path, label)):
            if file.endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(os.path.join(path, label, file))
                if img is not None:
                    imgs.append(cv2.resize(img, (size, size)))
                    lbls.append(label)
    return np.array(imgs), np.array(lbls)

X, y = load_images(DATA_PATH, IMG_SIZE)
X = X / 255.0
y_enc = to_categorical(LabelEncoder().fit_transform(y))
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(y_enc.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

aug = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.2, horizontal_flip=True
)
aug.fit(X_train)
model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"\n📊 Точность: {acc:.4f}, Потери: {loss:.4f}")

y_pred = model.predict(X_test)
y_pred_lbl = np.argmax(y_pred, axis=1)
y_true_lbl = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true_lbl, y_pred_lbl)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Предсказание"), plt.ylabel("Истинное")
plt.title("Матрица путаницы")
plt.show()

errors = np.where(y_pred_lbl != y_true_lbl)[0]
for idx in errors[:5]:
    plt.imshow(X_test[idx])
    plt.title(f"Истинное: {np.unique(y)[y_true_lbl[idx]]}, Предсказано: {np.unique(y)[y_pred_lbl[idx]]}")
    plt.axis('off')
    plt.show()
model.save("road_sign_model.h5")
from tensorflow.keras.models import load_model
model = load_model("road_sign_model.h5")

