# Semantic Segmantation for FloodNet Dataset

![23](https://github.com/user-attachments/assets/3320727f-4d88-4576-96e1-d80829d2f525)

## Overview

FloodNet is a semantic segmentation project designed to identify flooded and non-flooded regions in aerial imagery. The project uses deep learning techniques based on the DeepLabV3+ architecture using TensorFlow and Keras. to classify each pixel in an image into predefined categories (e.g., water, Tree, Vehicle).

This notebook implements the end-to-end pipeline for:

- **An EfficientNetV2-S backbone pretrained on ImageNet for feature extraction**.
- **Dilated Spatial Pyramid Pooling (ASPP) for capturing multiscale context**.
- **A custom decoder to upsample features to the input image resolution**.
- **Preparing and preprocessing data**.
- **Training and validating the segmentation model**.
- **Evaluating performance metrics (e.g., Dice Coefficient, IoU)**.
- **Performing inference on test images with visualized outputs**.

### Classes:
The dataset includes **10 classes**, each representing different features in flood-affected areas:

| Class ID | Class Name         |
|----------|--------------------|
| 0        | Background         |
| 1        | Building Flooded   |
| 2        | Building Non-Flooded |
| 3        | Road Flooded       |
| 4        | Road Non-Flooded   |
| 5        | Water              |
| 6        | Tree               |
| 7        | Vehicle            |
| 8        | Pool               |
| 9        | Grass              |


## Code Breakdown

### Input Specification
The model accepts images of a specific shape (shape) defined during initialization. The input layer:
```python
model_input = Input(shape=shape)
```
ensures compatibility with the EfficientNetV2 backbone.
### Backbone Network
EfficientNetV2-S is employed for extracting hierarchical feature maps:
```python
backbone = tf.keras.applications.EfficientNetV2S(include_top=False, weights="imagenet", input_tensor=model_input)
backbone.trainable = True
```
The backbone's key feature maps are accessed for multiscale processing:
- **block6b_expand_activation (smallest spatial resolution, richest features)**
- **block4b_expand_activation**
- **block3b_expand_activation**
- **block2b_expand_activation (largest spatial resolution)**

### Dilated Spatial Pyramid Pooling (ASPP)
For multiscale context, the ASPP block processes outputs from deeper layers:
```python
input_a = DilatedSpatialPyramidPooling(input_a, num_filters=256)
```
Each ASPP block includes parallel dilated convolutions with varying dilation rates.

### Upsampling and Decoder
Outputs from the ASPP and other backbone layers are upsampled to match the desired resolution:
```python
input_a = UpSampling2D(size=(16, 16), interpolation="bilinear")(input_a)
```
### Final Layer
The last layer applies a 1x1 convolution and softmax activation:
```python
outputs = Conv2D(num_classes, kernel_size=(1, 1), padding="valid", activation="softmax")(x)
```
This generates a segmentation map with probabilities across all classes.

### Training Hyperparameter:
- **Epochs**: 40
- **Batch Size**: 2
- **Optimizer**: AdamW with initial learning rate of 1e-3 and weight decay of 1e-5
- **Learning Rate Scheduler**: ReduceLROnPlate

#### Metrics include:
- Dice Coefficient: Measures overlap between predicted and true masks.
- Intersection over Union (IoU): Computes pixel-wise intersection vs union for each class.

### Loss and Metric Plots


<img width="1008" alt="Screenshot 2024-11-14 at 10 45 59 am" src="https://github.com/user-attachments/assets/4ddeaa5f-16ec-4109-a1a7-e77cca8de1f7">

### Inference and Visualization

<img width="1004" alt="prediction" src="https://github.com/user-attachments/assets/0bbc681b-59ed-4f1a-8418-e8c944b6f633">

### Submission 
<img width="1124" alt="Screenshot 2024-11-14 at 10 48 14 am" src="https://github.com/user-attachments/assets/8c9a7280-e945-483f-bbaf-7434fb51980f">

Submission
<img width="1124" alt="Screenshot 2024-11-14 at 10 48 14 am" src="https://github.com/user-attachments/assets/8c9a7280-e945-483f-bbaf-7434fb51980f">

