# Heart Disease Data Preprocessing and Feature Engineering using Apache Spark

## Project Overview
This project preprocesses a heart disease dataset using Apache Spark. The goal is to apply feature engineering techniques such as one-hot encoding, filtering, and quantization of cholesterol levels. The preprocessed dataset will be saved for further machine learning analysis.

## Steps

1. **Data Loading**: Load the heart disease dataset from a CSV file.
2. **One-Hot Encoding**: Convert the categorical variable `cp` (chest pain type) into a vector of binary values.
3. **Feature Derivation**: Add a new feature `powerOfTrestbps` by calculating the square of the `trestbps` (resting blood pressure).
4. **Data Filtering**: Filter patients who are older than 50 and have a resting blood pressure greater than 140.
5. **Quantization**: Create a new categorical column `cholesterol_level` based on `chol` values (Low, Medium, High).
6. **Data Reduction**: Count patients with high cholesterol.
7. **Data Export**: Save the preprocessed dataset into a CSV file.

## Folder Structure

heart_disease_spark_project/  
├── data/  
│   └── heart_disease_data.csv         
|  
│── main.py                        
│  
├── output/  
│   └── processed_heart_disease_data.csv           
├── README.md                            
├── requirements.txt                     
└── .gitignore                          

## Running the Project

1. Install the required dependencies: pip install -r requirements.txt

2. Run the preprocessing script: python main.py

3. Check the output in the `output/` folder.
