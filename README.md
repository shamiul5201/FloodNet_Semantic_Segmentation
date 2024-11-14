# Semantic_Segmentation_Application_FloodNet
Exploring the Environment with Semantic Segmentation: A FloodNet Application for Better Scene Understanding.

![23](https://github.com/user-attachments/assets/3320727f-4d88-4576-96e1-d80829d2f525)

## Description
This project tackles the challenge of classifying each pixel in flood-related images into 10 categories using the FloodNet dataset. By creating a model that accurately segments scenes in flood-affected areas, it aims to support a clearer understanding of environmental data, helping inform disaster response and future planning efforts

## About the dataset 
This project uses the FloodNet dataset for segmenting images, though it's too large to include directly in this repository.

### Dataset Details:
- **Training Set**: 1,843 images
- **Testing Set**: 500 images for evaluation
These samples cover a range of environmental and flood-related categories to help train and test the segmentation model.

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

### Segmentation Model 
To train the model, I used a convolutional block structure designed to capture detailed features in the images. Started with EfficientNetV2S as the backbone, using layers with different dilation rates to capture features at various scales. The feature maps were then upsampled and combined to create a richer representation of the image.

### Data augmentations
To help the model generalize better and avoid overfitting, I have applied data augmentation techniques. These included random horizontal and vertical flips, slight rotations and shifts, and adjustments to brightness and contrast. This added variation to the training images, helping the model learn a broader range of patterns.

```python
def transforms(self):
        
        train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, 
                               shift_limit=0.2, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5)
        ])
        return train_transforms
```

### Training Hyperparameter:
- **Epochs**: 40
- **Batch Size**: 2
- **Optimizer**: AdamW with initial learning rate of 1e-3 and weight decay of 1e-5
- **Learning Rate Scheduler**: ReduceLROnPlate

### Loss and Metric Plots

<img width="1008" alt="Screenshot 2024-11-14 at 10 45 59 am" src="https://github.com/user-attachments/assets/4ddeaa5f-16ec-4109-a1a7-e77cca8de1f7">

### Submission 
<img width="1124" alt="Screenshot 2024-11-14 at 10 48 14 am" src="https://github.com/user-attachments/assets/8c9a7280-e945-483f-bbaf-7434fb51980f">


