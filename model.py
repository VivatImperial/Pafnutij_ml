import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

def load_model_and_predict(pic, path='model'):
    my_image = load_img(pic, target_size=(224, 224))

    pafnutij = models.load_model(path)
    pafnutij.compile(loss="binary_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy", "FalseNegatives", "FalsePositives", "TrueNegatives", "TruePositives",
                              "Precision", "Recall"])

    my_image = img_to_array(my_image)
    my_image = np.expand_dims(my_image, axis=0)
    generator = ImageDataGenerator(rescale=1. / 255.,

                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   zoom_range=0.05,
                                   rotation_range=25)

    res = pafnutij.predict(generator.flow(my_image, batch_size=1))
    return 'Модель прогнозирует наличие опухоли на изображении с вероятностью: ' + str(round(res[0][0], 3))
