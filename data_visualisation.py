import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.utils import shuffle, resample
from sklearn.manifold import TSNE

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to preprocess the dataset
def preprocess_data(df, max_unique_values=10):
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill missing values with median for numeric columns
    
    # Handle non-numeric values
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass  # If conversion to float fails, keep the column as it is (assume it's string data)
    
    # Drop unnecessary columns with a high number of unique categorical values
    unnecessary_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_values = df[col].nunique()
            if unique_values > max_unique_values:
                unnecessary_columns.append(col)
    df = df.drop(columns=unnecessary_columns)
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if len(df[col].unique()) <= 2:  # Binary categorical variable
            df[col] = pd.factorize(df[col])[0]
        else:  # Categorical variable with more than 2 categories
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    print("preprocessed")
    return df

# Function to load the dataset from Excel or CSV
def load_dataset(file_path):
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Only .xlsx and .csv are supported.")

# Function to visualize data using t-SNE and save the plot
def visualize_data(df, target_column, save_path, title):
    features = df.drop(target_column, axis=1)
    target = df[target_column]
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    tsne_df = pd.DataFrame(tsne_results, columns=['Dim1', 'Dim2'])
    tsne_df['target'] = target

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Dim1', y='Dim2', hue='target', palette='viridis', data=tsne_df, legend='full')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    
    return tsne_results, tsne_df

# Function to create pie charts
def create_pie_charts(df, target_column, save_path_prefix):
    # Original class distribution
    plt.figure(figsize=(8, 6))
    df[target_column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    plt.title('Class Distribution in Original Data')
    plt.ylabel('')
    plt.savefig(f'{save_path_prefix}_class_distribution_original.png')
    plt.close()

# Function to detect and remove outliers using Isolation Forest
def detect_and_remove_outliers(df, target_column):
    features = df.drop(target_column, axis=1)
    
    if len(df) > 10000:
        df = subsample_data(df, target_column, n_samples=10000)
        features = df.drop(target_column, axis=1)
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['outlier'] = iso_forest.fit_predict(features)
    
    # Separate outliers and non-outliers
    outliers = df[df['outlier'] == -1]
    non_outliers = df[df['outlier'] == 1]

    cleaned_df = non_outliers.drop('outlier', axis=1)
    print("outliers")
    
    return cleaned_df, outliers, non_outliers

# Function to subsample data while maintaining class distribution
def subsample_data(df, target_column, n_samples=10000):
    df_subsampled = df.groupby(target_column, group_keys=False).apply(lambda x: x.sample(int(np.rint(n_samples * len(x) / len(df))))).sample(n_samples)
    return df_subsampled

# Function to create comparison pie charts
def create_comparison_pie_charts(non_outliers, outliers, target_column, save_path_prefix):
    for class_value in non_outliers[target_column].unique():
        class_non_outliers = non_outliers[non_outliers[target_column] == class_value]
        class_outliers = outliers[outliers[target_column] == class_value]

        sizes = [len(class_non_outliers), len(class_outliers)]
        labels = ['Valid Points', 'Outliers']
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62'])
        plt.title(f'Class {class_value}: Valid Points vs Outliers')
        plt.savefig(f'{save_path_prefix}_class_{class_value}_valid_vs_outliers.png')
        plt.close()

# Main function to run the entire process
def main():
    file_path = 'D:/Ml_Projects/Automatic model maker v2/classification_datasets/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset/NHANES_age_prediction.csv'  

    # Get the folder of the dataset and create 'graphs' folder
    dataset_folder = os.path.dirname(file_path)
    graphs_folder = os.path.join(dataset_folder, 'graphs')
    create_directory(graphs_folder)

    # Load the dataset
    df_0 = load_dataset(file_path)
    df = preprocess_data(df_0)

    # Assume target column is the last column
    target_column = df.columns[-1]

    # Visualize original data
    tsne_results, tsne_df = visualize_data(df, target_column, os.path.join(graphs_folder, 'tsne_original.png'), 't-SNE Visualization of Original Data')

    # Create pie chart for original data
    create_pie_charts(df, target_column, os.path.join(graphs_folder, 'original'))

    # Detect and remove outliers
    cleaned_df, outliers, non_outliers = detect_and_remove_outliers(df, target_column)

    # Visualize cleaned data
    visualize_data(cleaned_df, target_column, os.path.join(graphs_folder, 'tsne_cleaned.png'), 't-SNE Visualization of Cleaned Data')

    # Create pie chart for cleaned data
    plt.figure(figsize=(8, 6))
    cleaned_df[target_column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    plt.title('Class Distribution in Cleaned Data')
    plt.ylabel('')
    plt.savefig(os.path.join(graphs_folder, 'cleaned_class_distribution.png'))
    plt.close()

    # Create comparison pie charts for each class
    create_comparison_pie_charts(non_outliers, outliers, target_column, os.path.join(graphs_folder, 'comparison'))

if __name__ == "__main__":
    main()
