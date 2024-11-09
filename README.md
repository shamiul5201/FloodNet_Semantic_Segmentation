# Semantic_Segmentation_Application_FloodNet
A Deep Dive into Environmental Scene Understanding Using Semantic Segmentation

![23](https://github.com/user-attachments/assets/3320727f-4d88-4576-96e1-d80829d2f525)

## Table of Contents

* [Description](#description)
* [Dataset](#-dataset)
* [Dependencies](#dependencies)
* [Installation](#-installation)
* [Contributor Expectations](#-contributor-expectations)
* [Known Issues & Challenges](#-known-issues--challenges)
* [Conclusion](#-conclusion)


## Description
This project focused on the challenge of pixel-wise semantic segmentation using the FloodNet dataset. The task was to classify each pixel in an image into one of 10 specific categories, including various environmental and flood-related classes. The primary goal was to develop a model capable of accurately segmenting scenes in flood-impacted areas, aiding in the deeper understanding and analysis of environmental data.

The motivation behind this project stemmed from the need for precise, automated image analysis in environmental monitoring. By segmenting each image at the pixel level, this project aimed to provide insights into the structure and impact areas within flood-prone environments, potentially aiding disaster response efforts and future planning.

## 📊 Dataset

In this project, the **FloodNet dataset** is used for semantic segmentation. However, due to the large size of the dataset, it cannot be uploaded directly to this repository.

### 📈 Dataset Overview:
- **Training Samples**: 1843 images for training.
- **Testing Samples**: 500 images for evaluation/testing.

### 🏷️ Classes:
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

### 🗂️ Directory Structure:
- `train/images`: 1843 JPG images for training.
- `train/masks`: 1843 PNG masks for training.
- `test/images`: 500 JPG images for evaluation/testing.

### 🔗 Accessing the Dataset:
Due to the large size of the **FloodNet dataset**, it cannot be included in this repository. However, you can access and download the dataset from the official source:

- **FloodNet Dataset**: [FloodNet Dataset - Bina Lab, University of Maryland](https://maryam.is.umbc.edu/BinaLab/research.php)

Please ensure to follow the dataset's terms and conditions before using it.

---

This format is visually appealing and easy to follow, while still providing all the necessary details about the dataset, its structure, and where to find it. The use of icons like 📊, 📈, 🏷️, 🗂️, 🏆, and 🔗 helps to make the section engaging and organized.

---

## Dependencies

- **Core Libraries**:
  - **CUDA & Numba**: Accelerates computations using GPU.
  - **NumPy**: Efficient data manipulation and numerical operations.
  - **OpenCV**: Image processing capabilities for handling and transforming images.
  - **Albumentations**: Advanced image augmentations to enhance model generalization.
  - **PIL (Python Imaging Library)**: Additional image handling support.
  - **TensorFlow & Keras**: Deep learning framework for building and training segmentation models.
  - **Scikit-Learn**: Data splitting and other utilities for preparing training and test sets.
- **Additional Libraries**: 
  - **Requests**: Handles data download and API calls.
  - **Zipfile**: Manages compressed data files.

---

## 🚀 Installation

Follow these steps to set up the environment and start using this project:

1. **Clone the Repository**

   Start by cloning the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/Semantic_Segmentation_Application_FloodNet.git
   cd Semantic_Segmentation_Application_FloodNet

2. **Install Dependencies Manuallyy**

   Since this project is built using Jupyter Notebooks, there's no requirements.txt file. You can manually install the necessary libraries by running:
   ```bash
   pip install numba numpy matplotlib opencv-python albumentations Pillow tensorflow scikit-learn

3. **Verify Installations**

   After installing the dependencies, verify that the key libraries are working by running the following in a Python terminal:
   ```bash
   pip install numba numpy matplotlib opencv-python albumentations Pillow tensorflow scikit-learn

4. **Set Up GPU Support**

   If you’re using a compatible GPU, make sure CUDA is installed for faster computations. Numba and TensorFlow will automatically use the GPU if it's available.

5. **Optional: Launch Jupyter Notebook**

   To explore the project step-by-step using Jupyter Notebooks:
   ```bash
   jupyter notebook

---

## 🤝 Contributor Expectations

I welcome contributions to this project! To ensure smooth collaboration and maintain project quality, please follow these guidelines:

1. **Fork the Repository**  
   Start by forking this repository to your GitHub account. This allows you to freely make changes without affecting the original project.

2. **Create a New Branch**  
   Before making any changes, create a new branch based on the issue you're addressing. This helps keep your work organized and separate from the main branch.
   ```bash
   git checkout -b feature/your-feature-name
   
3. **Write Meaningful Commit Messages**
   When committing your changes, write clear and concise commit messages that describe what you’ve done. A good commit message might look like this:
    ```bash
   git commit -m "Fix bug in data preprocessing"
    
4. **Keep Changes Focused**
   Each pull request should focus on a specific change. Try to avoid mixing different types of changes (e.g., refactoring code and adding features) in the same PR.
   
6. **Be Respectful and Collaborative**
Collaboration and communication are key! Be respectful of other contributors, provide constructive feedback, and be open to discussion and suggestions.


## 🚧 Known Issues & Challenges

During the development of this project, there were several challenges faced, particularly with the **helper functions** and the **model architecture**. Below are the key issues:

### 🛠️ Helper Functions

One of the main challenges was working with the helper functions, which are essential for preprocessing, augmentation, and managing the dataset. Initially, I faced issues related to:
- **Inconsistent data formats**: Some of the images and masks were not in the expected format, which caused errors during loading and processing.
- **Inefficient augmentation**: The data augmentation process initially led to slow training times, especially when handling large datasets with high-resolution images.
  
After several iterations, I was able to fine-tune the helper functions to handle data more efficiently and improve the overall processing speed.

### 🤖 Model Architecture

Another major challenge was finding the right model for the task. I experimented with multiple **image segmentation models**, such as U-Net, FCN, and others, but faced some key issues:
- **Model performance**: Many of the models showed poor performance on certain classes, such as distinguishing flooded areas from non-flooded areas, which led to low accuracy and high loss rates.
- **Training instability**: The model training would sometimes become unstable due to overfitting or underfitting, especially with certain architectures.

After trying different approaches, I settled on a custom model architecture that combines **ResNet50** as the backbone for feature extraction, paired with upsampling and skip connections to help preserve spatial information. This approach led to better performance and a more stable training process.

---

These challenges were crucial in shaping the final model and the overall workflow of the project. They provided valuable learning experiences in model selection, data preprocessing, and performance optimization. 

---

## 🏁 Conclusion

This project has been an insightful journey into **semantic segmentation** using the **FloodNet dataset**. Through experimenting with various **models** and **helper functions**, I was able to develop a working model that accurately segments flood-related features from high-resolution aerial images.

### Key Takeaways:
- **Model Development**: After several trials with different segmentation models, I successfully built a custom architecture based on **ResNet50**, which improved both the model’s stability and its accuracy.
- **Data Processing**: By fine-tuning the **data augmentation** and **preprocessing** functions, I was able to handle large image datasets more efficiently.
- **Challenges Overcome**: This project reinforced the importance of perseverance in experimenting with different approaches to solve complex problems, especially in computer vision tasks.

### Future Directions:
- **Model Optimization**: There is still room to improve the model’s accuracy, especially on certain flood-related classes. Future work could involve experimenting with **advanced architectures** or **fine-tuning** hyperparameters.
- **Dataset Expansion**: Incorporating more diverse images or using data from other sources could improve the generalization of the model.
- **Real-World Application**: This segmentation model can potentially be applied to real-world flood monitoring systems, where it could assist in identifying flood-prone areas and post-flood damage.

Overall, this project provides a solid foundation for future improvements and applications in **flood detection** and **disaster management**. The knowledge gained from this project has equipped me with a deeper understanding of image segmentation and the challenges that come with it.


