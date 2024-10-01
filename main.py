import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from preprocessing_data import load_data
from train_model import train_model

def plot_classification_report(report_df, output_folder):
    # Extract values from the report DataFrame
    labels = report_df.index[:-1]  # Exclude 'accuracy'
    precision = report_df['precision'][:-1]
    recall = report_df['recall'][:-1]
    f1_score = report_df['f1-score'][:-1]

    x = range(len(labels))  # The x locations for the groups

    # Create the bar chart
    width = 0.2
    plt.bar(x, precision, width=width, label='Precision', color='b', align='center')
    plt.bar([p + width for p in x], recall, width=width, label='Recall', color='g', align='center')
    plt.bar([p + 2 * width for p in x], f1_score, width=width, label='F1 Score', color='r', align='center')

    # Add labels and title
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Classification Report')
    plt.xticks([p + width for p in x], labels)
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'classification_report_plot.png'))
    plt.show()

def main():
    # Load the data
    df = load_data()

    # Print the data types of each column
    print(df.dtypes)

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Prepare features and target variable
    X = df.drop('salary', axis=1)  # Features
    y = df['salary']  # Target variable

    # Train the model
    model = train_model(X, y)

    # Check if model is trained
    if model is None:
        print("Model training failed!")
        return

    # Generate predictions
    y_pred = model.predict(X)

    # Generate classification report
    report = classification_report(y, y_pred, output_dict=True)

    # Convert report to DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Define output folder and ensure it exists
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    # Save the classification report as a CSV
    report_df.to_csv(os.path.join(output_folder, 'classification_report.csv'))

    # Print the classification report
    print(report_df)

    # Plot and save the classification report visualization
    plot_classification_report(report_df, output_folder)

if __name__ == '__main__':
    main()




