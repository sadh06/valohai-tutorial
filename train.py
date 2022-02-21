import numpy as np
import tensorflow as tf
import valohai


input_path = 'mnist.npz'
with np.load(input_path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])


optimizer = tf.keras.optimizers.Adam(learning_rate=valohai.parameters('learning_rate').value)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,

            loss=loss_fn,
            metrics=['accuracy'])


model.fit(x_train, y_train, epochs=valohai.parameters('epoch').value)


model.evaluate(x_test,  y_test, verbose=2)

output_path = valohai.outputs().path('model.h5')
model.save(output_path)