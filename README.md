# AI4AFP - AntiFungal Peptide Prediction in AI

## Project Overview
AI4AFP (Artificial Intelligence for Anti-Fungal Peptides) is an open-source platform leveraging artificial intelligence to predict antibiotic resistance. The project applies machine learning techniques to analyze pathogen genomic data, focusing on foodborne pathogens and their resistance to various antibiotics.

## Key Features
1. **Sequence Encoding**: Implements advanced natural language processing techniques, including Doc2Vec and BERT, to transform genomic sequences into machine-readable vector representations.
2. **Multi-model Prediction**: Integrates multiple machine learning algorithms (such as Logistic Regression, Random Forest, and XGBoost) to enhance prediction accuracy.
3. **Cross-validation**: Utilizes K-fold cross-validation to ensure model robustness and generalization.
4. **Model Ensemble**: Aggregates predictions from multiple models to improve overall performance.

## Technical Architecture
- **Data Preprocessing**: Cleansing, standardizing, and extracting features from genomic sequences.
- **Sequence Encoding Methods**:
  - *Doc2Vec*: Converts gene sequences into vectors by treating them as text documents.
  - *BERT*: Employs deep learning to capture contextual relationships in sequences.
- **Machine Learning Models**:
  - *Logistic Regression*: A fundamental classification model.
  - *Random Forest*: An ensemble learning method based on decision trees.
  - *XGBoost*: A highly efficient gradient boosting tree model.
- **Evaluation Metrics**: Uses K-fold cross-validation, confusion matrices, ROC curves, and precision-recall analysis.

## Use Cases
- Rapid antibiotic resistance prediction in clinical microbiology labs.
- Detection of resistant strains in food safety monitoring.
- Public health surveillance and outbreak tracking.
- Antibiotic resistance research and drug development.

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





