import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Dot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix)
import seaborn as sns

# ===================== Hyper‑parameters (UPDATED) ===================== #
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4          # 0.0001
EMBEDDING_DIM = 256           # ↑ from 128 to 256
FIRST_LAYER_UNITS = 256       # ↓ from 512 to 256
SECOND_LAYER_UNITS = 128      # ↓ from 256 to 128
FIRST_DROPOUT_RATE = 0.4      # ↑ from 0.3 to 0.4
SECOND_DROPOUT_RATE = 0.2     # unchanged
# ===================================================================== #

# ---------------------- Data Loading Helpers ------------------------- #

def load_data(path):
    data = np.load(path, allow_pickle=True)
    return data['drugA'], data['drugB'], data['labels']

train_A, train_B, train_labels = load_data('../dataset_double_tower_model/train_data.npz')
test_A, test_B, test_labels   = load_data('../dataset_double_tower_model/test_data.npz')

train_dataset = (
    tf.data.Dataset.from_tensor_slices(({'input_A': train_A, 'input_B': train_B}, train_labels))
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(({'input_A': test_A, 'input_B': test_B}, test_labels))
    .batch(BATCH_SIZE)
)

# ---------------------- Siamese Model Definition --------------------- #

def create_siamese_network(input_dim: int) -> Model:
    """Builds the twin‑tower network with shared weights."""

    def tower_block():
        return tf.keras.Sequential([
            Dense(FIRST_LAYER_UNITS, activation='relu', input_shape=(input_dim,)),
            Dropout(FIRST_DROPOUT_RATE),
            Dense(SECOND_LAYER_UNITS, activation='relu'),
            Dropout(SECOND_DROPOUT_RATE),
            Dense(EMBEDDING_DIM, activation=None)
        ])

    inp_A = Input(shape=(input_dim,), name='input_A')
    inp_B = Input(shape=(input_dim,), name='input_B')

    tower = tower_block()
    emb_A = tower(inp_A)
    emb_B = tower(inp_B)

    cosine_sim = Dot(axes=1, normalize=True)([emb_A, emb_B])
    output = Dense(1, activation='sigmoid')(cosine_sim)

    return Model(inputs=[inp_A, inp_B], outputs=output)

model = create_siamese_network(input_dim=train_A.shape[1])
model.summary()

# ------------------------- Compilation ------------------------------- #
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# --------------------------- Training -------------------------------- #
callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=[callback]
)

# ------------------------- Visualisation ----------------------------- #
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
plt.tight_layout(); plt.show()

# -------------------------- Evaluation ------------------------------- #
true_labels, probs = [], []
for batch in test_dataset:
    x, y = batch
    true_labels.extend(y.numpy())
    probs.extend(model.predict(x, verbose=0).flatten())

y_true = np.array(true_labels)
probs  = np.array(probs)
y_pred = (probs >= 0.5).astype(int)

print('\n=== Test Metrics ===')
print(f'Accuracy : {accuracy_score(y_true, y_pred):.4f}')
print(f'Precision: {precision_score(y_true, y_pred):.4f}')
print(f'Recall   : {recall_score(y_true, y_pred):.4f}')
print(f'F1 Score : {f1_score(y_true, y_pred):.4f}')
print(f'AUC      : {roc_auc_score(y_true, probs):.4f}')

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Interaction', 'Interaction'],
            yticklabels=['No Interaction', 'Interaction'])
plt.ylabel('True Label'); plt.xlabel('Predicted Label'); plt.title('Confusion Matrix'); plt.show()

model.save('drug_interaction_model.keras')
print('Model saved as drug_interaction_model.keras')
