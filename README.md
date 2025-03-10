# AI4AFP - AntiFungal Peptide (AFP) Prediction in AI

## Project Overview
AI4AFP (Artificial Intelligence for Anti-Fungal Peptides) is an open-source toolbox leveraging artificial intelligence to predict the function of peptides wuth anti-fungal activity. The project applies machine learning techniques to analyze candidate peptides. To expedite the discovery process for effective AFPs, we aim to create a predictive model utilizing computational methods and extensive antifungal peptide resources.

## Key Features
1. **Sequence Encoding**: Implements advanced natural language processing techniques, including Doc2Vec and BERT with our previous approach, [PC6](https://github.com/wccheng1210/AI4AFP/blob/main/PC6_encoding.py), to transform peptide sequences into machine-readable vector representations.
2. **Multi-model Prediction**: Integrates multiple machine learning algorithms (such as SVM, Random Forest, and CNN) to enhance prediction accuracy.
3. **Cross-validation**: Utilizes K-fold cross-validation to ensure model robustness and generalization.
4. **Model Ensemble**: Aggregates predictions from multiple models to improve overall performance.

## Technical Architecture
- **Data Preprocessing**: Cleansing, standardizing, and extracting features from peptide sequences.
- **Sequence Encoding Methods**:
  - *Doc2Vec*: Converts peptide sequences into vectors by treating them as text documents.
  - *BERT*: Employs deep learning to capture contextual relationships in sequences.
  - *PC6*: A physicochemical-based embedding approach for feature representation.
- **Machine Learning Models**:
  -*SVM (Support Vector Machine)*: A robust classification algorithm.
  -*Random Forest*: An ensemble learning method based on decision trees.
  -*CNN (Convolutional Neural Network)*: A deep learning model for feature extraction and classification.
- **Evaluation Metrics**: Uses K-fold cross-validation, confusion matrices, ROC curves, and precision-recall analysis.


## Installation and Usage
The project is hosted on GitHub ([https://github.com/lsbnb/AI4AFP](https://github.com/lsbnb/AI4AFP)), where users can clone the repository and follow setup instructions. The repository primarily consists of Python scripts and Jupyter Notebooks, facilitating model training and evaluation for researchers.

## Demo Website
A live demo of AI4AFP is available at: [https://axp.iis.sinica.edu.tw/AI4AFP](https://axp.iis.sinica.edu.tw/AI4AFP)

## License Information
AI4AFP is licensed under a standard Creative Commons (CC) license, allowing users to:
- **Share**: Copy and redistribute the material.
- **Adapt**: Modify, transform, and build upon the work.

Specific license terms may include attribution requirements, non-commercial use restrictions, or share-alike conditions. Users should review the project documentation for detailed licensing terms.

## Future Development Directions
- Developing more advanced deep learning architectures.
- Enhancing user interface accessibility.
- Extending coverage to additional pathogens and antibiotics.





