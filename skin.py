import glob
import warnings

import matplotlib.pyplot as plt
import neptune
import tensorflow as tf
import tensorflow_hub as hub
from neptune.integrations.tensorflow_keras import NeptuneCallback
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.models import Sequential
from tensorflow_addons.metrics import F1Score

tf.config.run_functions_eagerly(True)

warnings.filterwarnings("ignore")

plt.rcParams['font.size'] = 10

fpath = r'C:\Users\Lawaiian\PycharmProjects\dlmi\data'
# fpath = r'C:\Users\Lawaiian\Desktop\CS\Data Science 2\deadlock\tf_dogs_xiaruize\data\Images'
random_seed = 42

img_size = 224
batch_size = 32
train = tf.keras.utils.image_dataset_from_directory(
    fpath,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical",
)

val = tf.keras.utils.image_dataset_from_directory(
    fpath,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical"
)

run = neptune.init_run(
    project="lawaiian999/DLMI",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTMwOWM4Ny1jNGM5LTQ4NTEtOTFlNS0wN2Q3ZjA0ZGVlMTUifQ==",
)  # your credentials

# Prepare params
parameters = {'dense_units': 128,
              'activation': 'relu',
              'dropout': 0.23,
              'learning_rate': 0.15,
              'batch_size': 32,
              'n_epochs': 30}

run['model/params'] = parameters

class_names = train.class_names

Model_URL = 'https://kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/50-classification/versions/2'

# Prepare model
model = Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_size, img_size, 3)),
    hub.KerasLayer(Model_URL),
    tf.keras.layers.Dropout(0.23),  # 根据需要调整丢弃率
    tf.keras.layers.Dense(7, activation="softmax")])


optimizer = tf.keras.optimizers.SGD(learning_rate=parameters['learning_rate'])

model.compile(
    # loss=tf.keras.losses.CategoricalCrossentropy(),
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=["accuracy",
             Precision(name='precision'),
             Recall(name='recall'),
             F1Score(num_classes=7, average='weighted', name='f1_score'),
             AUC(num_thresholds=200,
                 curve='ROC',
                 summation_method='interpolation',
                 name=None,
                 dtype=None,
                 thresholds=None,
                 multi_label=False,
                 num_labels=None,
                 label_weights=None,
                 from_logits=False)
             ]
)
model.build((img_size, img_size, 3))
# Log model summary
model.summary(print_fn=lambda x: run['model/summary'].log(x))
# Train model
neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

history = model.fit(train,
                    batch_size=parameters['batch_size'],
                    epochs=parameters['n_epochs'],
                    validation_data=val,
                    callbacks=[neptune_cbk])

# Log model weights
model.save('trained_model')
model.save('dlmi_3_model.h5')
run['model/weights/saved_model'].upload('trained_model/saved_model.pb')
for name in glob.glob('trained_model/variables/*'):
    run[name].upload(name)

# Evaluate model
eval_metrics = model.evaluate(val, verbose=0)
for j, metric in enumerate(eval_metrics):
    run['test/scores/{}'.format(model.metrics_names[j])] = metric


plt.figure(figsize=(20, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('model f1_score')
plt.ylabel('f1_score')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
