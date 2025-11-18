Of course, here is a README for your GitHub project.

---

# Decision Tree Classifier from Scratch

This project is a Python implementation of a decision tree learning algorithm from scratch. It is designed to classify data based on the popular "Car Evaluation" dataset. The script loads the data, builds a decision tree, evaluates its performance, and visualizes the learning process.

## Key Features

*   **Custom Decision Tree Implementation**: The core of the project is a from-scratch implementation of a decision tree, using concepts like Entropy and Information Gain.
*   **Data Handling**: The script uses the `pandas` library to efficiently load and manipulate the dataset.
*   **Model Training and Testing**: The dataset is split into training and testing sets to evaluate the model's performance on unseen data.
*   **Performance Metrics**: To evaluate the model, the script calculates and displays key metrics, including:
    *   Accuracy
    *   Precision
    *   Recall
    *   F1-Score
*   **Learning Curve Visualization**: A learning curve is plotted using `matplotlib` to show how the model's performance changes with the size of the training data.

## How It Works

The algorithm follows these steps:

1.  **Load Data**: The `car.csv` dataset is loaded into a pandas DataFrame.
2.  **Split Data**: The data is divided into a training set and a testing set.
3.  **Build Tree**: A decision tree is recursively built from the training data using the following logic:
    *   **Entropy** is calculated to measure the impurity of a dataset.
    *   **Information Gain** is used to determine the best feature to split the data at each node.
    *   The tree stops growing when a node is "pure" (contains data from only one class) or when there are no more features to split on.
4.  **Make Predictions**: The trained tree is used to predict the class labels for the test set.
5.  **Evaluate Performance**: The predicted labels are compared to the actual labels from the test set to calculate performance metrics.
6.  **Plot Learning Curve**: The model is trained on increasing subsets of the training data, and the accuracy on the test set is plotted to visualize how more data impacts performance.

## Getting Started

### Prerequisites

*   Python 3.x
*   The following Python libraries are required:
    *   pandas
    *   numpy
    *   matplotlib

### Installation

1.  Clone the repository:
    ```bash
    git clone (https://github.com/LachlanMayes/Learning-Decision-Tree-)
    ```

2.  Install the required libraries using pip:
    ```bash
    pip install pandas numpy matplotlib
    ```

3.  Download the "Car Evaluation" dataset and save it as `car.csv` in the same directory as the script. You can find the dataset [here](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation).

### Running the Script

To run the program, simply execute the `LDT.py` file from your terminal:

```bash
python LDT.py```

## Expected Output

When you run the script, you will see the following in your console:

1.  The sizes of the training and testing sets.
2.  A detailed classification report with precision, recall, and F1-score for each class, as well as macro and weighted averages.
3.  The progress of the learning curve generation, showing the test accuracy at different training set sizes.

Finally, a window will pop up displaying the learning curve plot, which visualizes the model's accuracy as the number of training examples increases.
