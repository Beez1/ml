# Machine Learning Assignment 2: Soil Type Classification Using Satellite Imagery

This repository contains the implementation of a Machine Learning assignment with two main components:

## Task 1: Literature Review (30%)
A comprehensive literature review on soil type identification using satellite imagery, focusing on:
- Recent approaches (2019-2025)
- Comparison of methods
- Data sources and their characteristics
- Performance metrics and results
- Future directions in the field

## Task 2: CNN-Based Classification (70%)
Implementation of a Convolutional Neural Network (CNN) for classifying land cover/soil types from satellite imagery.

### Dataset
The dataset consists of satellite imagery with the following classes:
- AnnualCrop (3000 images)
- Forest (3000 images)
- HerbaceousVegetation (3000 images)
- Highway (2500 images)
- Industrial (2500 images)
- Pasture (2000 images)
- PermanentCrop (2500 images)
- Residential (3000 images)
- River (2500 images)
- SeaLake (3000 images)

All images are 64x64 RGB patches extracted from Sentinel-2 satellite imagery (derived from the EuroSAT dataset).

### Project Structure
- `doc/` - Documentation for both tasks
  - `Task1_LiteratureReview.txt` - Literature review on soil type identification
  - `Task2_CNN_Report.txt` - Report on the CNN implementation
- `notebook/` - Jupyter notebook implementation
  - `Task2_CNN_Notebook.ipynb` - CNN model implementation and analysis
- `Dataset/` - Contains all the satellite imagery (27,000 images)
- `final_submission/` - PDF versions of documentation for submission
- `saved_model/` - Trained model weights and architecture (after running the notebook)

### Requirements
- Python 3.8+
- TensorFlow 2.8
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL/Pillow
- Seaborn
- tqdm

### Setup and Execution

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment (optional but recommended)
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Running the Notebook**
   ```bash
   # Start Jupyter notebook
   jupyter notebook notebook/Task2_CNN_Notebook.ipynb
   ```
   
   The notebook is structured to:
   - Load and preprocess the dataset
   - Define the CNN architecture (3 convolutional blocks with batch normalization)
   - Train the model with data augmentation and early stopping
   - Evaluate performance on the test set
   - Visualize results through confusion matrices and feature maps
   - Save the trained model

3. **Expected Results**
   - Overall accuracy: ~90.8%
   - Best performing classes: Forest (95.9% F1) and SeaLake (96.0% F1)
   - Training time: ~15 minutes on GPU, ~1 hour on CPU

4. **Viewing Documentation**
   - The full reports are available in the `final_submission/` directory:
     - `Task1_LiteratureReview.pdf` - Literature review
     - `Task2_CNN_Report.pdf` - Detailed CNN implementation report

## Acknowledgments
- EuroSAT dataset: Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019)
- All references are listed in the respective documentation files 