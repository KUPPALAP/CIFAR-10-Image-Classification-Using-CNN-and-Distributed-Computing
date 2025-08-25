# CIFAR-10 Image Classification using CNN and Distributed Computing

##  Project Overview
This project implements a **Convolutional Neural Network (CNN)** for **CIFAR-10 image classification** and integrates **Apache Spark** for distributed training. The focus is on building a scalable pipeline to classify 10 categories (e.g., airplane, automobile, bird, cat, etc.) and optimize training efficiency.

Key Contributions:
- Applied CNN to CIFAR-10 dataset with 60,000 images (32x32 RGB).
- Designed a multi-layer CNN with dropout for regularization.
- Implemented distributed computing with **Apache Spark** across virtual machines.
- Achieved **70–75% test accuracy** with good generalization.


## Project Structure
- `final_project_code_cnn_.ipynb` → Jupyter Notebook with CNN implementation, training, and evaluation.
- `big data ppt (2).pptx` → Project presentation summarizing objectives, architecture, and results.
- `README.md` → This documentation file.

 

##  Installation & Setup

### Requirements
- Python 3.8+
- Jupyter Notebook
- Libraries:
  ```bash
  pip install tensorflow keras pyspark pandas numpy matplotlib seaborn
  ```

 

##  Running the Project
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run the notebook `final_project_code_cnn_.ipynb`.
3. Steps performed:
   - Preprocess CIFAR-10 images (normalization, one-hot encoding).
   - Train CNN with convolutional, pooling, dropout, and dense layers.
   - Use Spark to distribute data across executors for parallel training.
   - Evaluate model performance with accuracy, loss graphs, and confusion matrix.

 

##  CNN Architecture
- **Conv Layer 1:** 32 filters, 3x3 kernel, ReLU + MaxPooling
- **Conv Layer 2:** 64 filters, 3x3 kernel, ReLU + MaxPooling
- **Conv Layer 3:** 128 filters, 3x3 kernel, ReLU + MaxPooling
- **Flatten Layer**
- **Dense Layer:** 128 units, ReLU + Dropout (0.5)
- **Output Layer:** 10 units, Softmax

 

##  Results
- **Accuracy:** ~70–75% on validation data.
- **Training vs Validation Accuracy:** Validation accuracy consistently higher → good generalization.
- **Loss:** Both training and validation loss decrease steadily, indicating stable convergence.
- **Confusion Matrix:** Shows class-level performance and areas for improvement.

 

##  Challenges & Future Work
- Improve accuracy using **Transfer Learning** (e.g., ResNet, VGG).
- Add **Grad-CAM explainability** for model decisions.
- Use advanced hyperparameter tuning methods (Bayesian optimization).
- Scale Spark configuration for larger datasets.

 

##  Team
-Karunakar Uppalapati
-Varshitha Reddy Davarapalli
- Saketha Kusu


