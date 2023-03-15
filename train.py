import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from shapely.geometry import mapping
import tensorflow as tf
import geopandas as gpd
from typing import List, Tuple

AERIAL_IMAGE_PATH = 'T36UXV_20200406T083559_TCI_10m.jp2'
LABEL_FILE_PATH = "./masks/Masks_T36UXV_20190427.shx"

EPOCHS = 20
BATCH_SIZE = 32

def main():
    all_features, all_labels = get_features_and_labels()

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),   # 32x32 -> 64x64
    ]

    model = unet_model(up_stack, down_stack, output_channels=2)

    model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.fit(all_features, all_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)



def get_features_and_labels() -> Tuple[List[np.array], List[np.array]]:
    all_features = []
    all_labels = []

    with rasterio.open(AERIAL_IMAGE_PATH, "r") as aerial_image_reader:
        meta = aerial_image_reader.meta
        aerial_image = reshape_as_image(aerial_image_reader.read())

    label_df = gpd.read_file(LABEL_FILE_PATH)
    label_df = label_df.to_crs({'init': meta['crs']['init']})
    label_df = label_df[~label_df['geometry'].isnull()]

    shapes = [mapping(row['geometry']) for num, row in label_df.iterrows()]

    src = rasterio.open(AERIAL_IMAGE_PATH, "r")
    masked_image, out_transform = rasterio.mask.mask(src, shapes, crop=False, nodata=0)
    masked_image = reshape_as_image(masked_image)
    mask = masked_image != 0
    mask = mask[:, :, 0] | mask[:, :, 1] | mask[:, :, 2]

    WIDTH = 128
    HEIGHT = 128
    for i in range(aerial_image.shape[0] // HEIGHT):
        for j in range(aerial_image.shape[0] // WIDTH):
            top = i * HEIGHT
            bottom = (i + 1) * HEIGHT
            left = j * WIDTH
            right = (j + 1) * WIDTH
            all_features.append(aerial_image[top:bottom, left:right])
            all_labels.append(mask[top:bottom, left:right])

    return all_features, all_labels


def upsample(filters, size, apply_dropout=False):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
        filters: number of filters
        size: filter size
        apply_dropout: If True, adds the dropout layer
    Returns:
        Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def unet_model(up_stack, down_stack, output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    main()
