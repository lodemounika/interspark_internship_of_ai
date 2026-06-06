import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Class names
class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

print("Training Images Shape:", X_train.shape)
print("Testing Images Shape :", X_test.shape)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    zoom_range=0.1
)

plt.figure(figsize=(10,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(X_train[i])

    plt.title(class_names[y_train[i][0]])

    plt.axis('off')

plt.tight_layout()

plt.show()

base_model = MobileNetV2(

    input_shape=(96, 96, 3),

    include_top=False,

    weights='imagenet'
)

# Freeze pretrained layers
base_model.trainable = False

model = models.Sequential([

    # Resize images dynamically
    layers.Resizing(96, 96),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation='relu'),

    layers.Dropout(0.4),

    layers.Dense(128, activation='relu'),

    layers.Dropout(0.3),

    layers.Dense(10, activation='softmax')
])

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)

# Display model summary
model.summary()

history = model.fit(

    datagen.flow(
        X_train,
        y_train,
        batch_size=64
    ),

    epochs=10,

    validation_data=(X_test, y_test)
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("\n===================================")
print("TEST ACCURACY :", round(test_accuracy, 4))
print("TEST LOSS     :", round(test_loss, 4))
print("===================================")

y_pred_probs = model.predict(X_test)

y_pred = np.argmax(y_pred_probs, axis=1)

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(classification_report(

    y_test,

    y_pred,

    target_names=class_names
))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12,10))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues',

    xticklabels=class_names,

    yticklabels=class_names
)

plt.xlabel("Predicted Labels")

plt.ylabel("True Labels")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.show()

plt.figure(figsize=(8,6))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Training vs Validation Accuracy')

plt.legend()

plt.grid(True)

plt.show()

plt.figure(figsize=(8,6))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Training vs Validation Loss')

plt.legend()

plt.grid(True)

plt.show()

model.save("cifar10_model.keras")

print("\nModel Saved Successfully!")

sample_image = X_test[0]

plt.figure(figsize=(4,4))

plt.imshow(sample_image)

plt.title("Test Image")

plt.axis('off')

plt.show()

# Predict image

prediction = model.predict(

    np.expand_dims(sample_image, axis=0)
)

predicted_class = np.argmax(prediction)

print("\nPredicted Class:",
      class_names[predicted_class])

print("Actual Class   :",
      class_names[y_test[0][0]])

loaded_model = tf.keras.models.load_model(
    "cifar10_model.keras"
)

print("\nSaved Model Loaded Successfully!")

print("\n===================================")
print("USER IMAGE TESTING")
print("===================================")

while True:

    index = input(
        "\nEnter image index between 0 to 9999 (or type 'exit'): "
    )

    if index.lower() == 'exit':

        print("Exiting User Testing...")

        break

    index = int(index)

    # Display image
    plt.figure(figsize=(4,4))

    plt.imshow(X_test[index])

    plt.title("Selected Test Image")

    plt.axis('off')

    plt.show()

    # Predict image
    prediction = model.predict(

        np.expand_dims(X_test[index], axis=0)
    )

    predicted_class = np.argmax(prediction)

    actual_class = y_test[index][0]

    print("\nPredicted Class :",
          class_names[predicted_class])

    print("Actual Class    :",
          class_names[actual_class])



