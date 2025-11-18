import pandas as panda
import math
import numpy as np
import matplotlib.pyplot as ploting

# class Node object definition it was easier to  implement and to understand than as a non object
class Node:
    # Node constructor
    def __init__(self, feature = None, prediction = None):
        self.feature = feature
        self.prediction = prediction
        self.children = {}

    # Adding children nodes (e.g possible answer"low") to the main Node (e.g Feature "Saftey") 
    def add_child(self, value, child_node):
        self.children[value] = child_node

    # check to see if the node is A prediction (Answer) or A decision 
    def is_leaf(self):
        return not self.children

#loads the file car.csv
def load_data(data):
    file = panda.read_csv(data)
    return file 

#splits the file into a test set and a train set and returns both 
def split_data(data,split = 0.5, seed = 42):
    train_set = data.sample(frac = split, random_state = seed)
    test_set = data.drop(train_set.index)
    
    return train_set, test_set

# calculates the entropy of the given data either the test_set or the train_set
def calculate_entropy(data):
    if data.empty:
        return 0
    target_column = data.columns[-1]
    class_count = data[target_column].value_counts()
    row_total = len(data)
    entropy = 0

    for count in class_count:
        probability = count / row_total
        entropy -= probability * math.log2(probability)
    
    return entropy

"""Takes that entrophy score from the parent node and the weighted child entrophy 
 and calcultes the information gained from the result of subtracting them both  """
def calculate_information_gain(data, feature):
    parent_entropy = calculate_entropy(data)
    unique_values = data[feature].unique()
    weighted_child_entropy = 0

    for value in unique_values:
        subset = data[data[feature] == value]
        subset_entropy = calculate_entropy(subset)
        weight = len(subset) / len(data)
        weighted_child_entropy += weight * subset_entropy
    
    information_gained = parent_entropy - weighted_child_entropy
    return information_gained

#Function finds the feature with the best infroamtion gained score by looping through ever feature 
def find_best_feature(data,features):
    best_feature = None
    max_info_gain = -1

    for feature in features:
        info_gained = calculate_information_gain(data, feature)

        if info_gained > max_info_gain:
            max_info_gain = info_gained
            best_feature = feature
    
    return best_feature

"""function builds the decison tree based off 3 situations 1. data is already pure
    2. no more features to test 3. checks features choose best one no information gained means
    the data is pure so return the tree else make sub trees on the sub data """
def build_tree(data, features):
    if data[data.columns[-1]].nunique() == 1:
        prediction = data[data.columns[-1]].iloc[0]
        return Node(prediction=prediction)
    
    elif len(features) == 0:
        majority_class = data[data.columns[-1]].mode()[0]
        return Node(prediction=majority_class)
    
    else:
        best_feature = find_best_feature(data, features)
        if calculate_information_gain(data, best_feature) == 0:
            majority_class = data[data.columns[-1]].mode()[0]
            return Node(prediction=majority_class)

        tree_node = Node(feature=best_feature)
        remaining_features = [f for f in features if f != best_feature]

        for value in data[best_feature].unique():
            subset_data = data[data[best_feature] == value]
            child_tree = build_tree(subset_data, remaining_features)
            tree_node.add_child(value, child_tree)
        
        return tree_node

""" Goes down the tree built from build_tree function best case goes deeper down the tree finding
new information on each node wrost case goes only one level deep and has to make a guess
"""
def predict(node, data_row):
    if node.is_leaf():
        return node.prediction
    
    feature_value = data_row[node.feature]
    if feature_value not in node.children:
        for child in node.children.values():
            if child.is_leaf():
                return child.prediction
        
        return predict(list(node.children.values())[0], data_row)
    
    next_node = node.children[feature_value]
    return predict(next_node, data_row)

# Helps the predict collect the answers as it goes down the tree for each data row
def make_predict(tree, data):
    data_to_predict = data.iloc[:, :-1]

    predictions = []
    for index, row in data_to_predict.iterrows():
        prediction = predict(tree, row)
        predictions.append(prediction)
        
    return predictions

"""Calculates precison, recall and F1 score for our LDT to see how well it worked on the given data
and prints the results  """
def calculate_and_print_metrics(y_true, y_pred, class_labels):
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Total Accuracy: {accuracy:.4f}\n")

    precision = {}
    recall = {}
    f1_score = {}

    labels = sorted(class_labels)

    for label in labels:
        tp = 0 
        fp = 0  
        fn = 0

        for true, pred in zip(y_true, y_pred):
            if true == label and pred == label:
                tp += 1
            elif true != label and pred == label:
                fp += 1
            elif true == label and pred != label:
                fn += 1

        precision[label] = tp / (tp + fp + 1e-7)
        recall[label] = tp / (tp + fn + 1e-7)
        f1_score[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label] + 1e-7)

    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 55)
    for label in labels:
        print(f"{label:<15} {precision[label]:<12.2f} {recall[label]:<12.2f} {f1_score[label]:<12.2f}")
    
    print("-" * 55)

    macro_precision = sum(precision.values()) / len(labels)
    macro_recall = sum(recall.values()) / len(labels)
    macro_f1 = sum(f1_score.values()) / len(labels)

    print(f"{'Macro Avg':<15} {macro_precision:<12.2f} {macro_recall:<12.2f} {macro_f1:<12.2f}")

    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    for label in labels:
        support = sum(1 for true in y_true if true == label)
        weighted_precision += precision[label] * support
        weighted_recall += recall[label] * support
        weighted_f1 += f1_score[label] * support

    weighted_precision /= total_predictions
    weighted_recall /= total_predictions
    weighted_f1 /= total_predictions

    print(f"{'Weighted Avg':<15} {weighted_precision:<12.2f} {weighted_recall:<12.2f} {weighted_f1:<12.2f}")

""" makes the ploting curve chart showing the results of the model to see how well it does with the 
data given if it pleatus early the model can't capture the patterns else if goes up near the end 
the model wants more data to learn from  """
def plot_learning_curve(train_data, test_data, training_percentages=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    training_sizes = []
    test_accuracies = []
    features = list(train_data.columns[:-1])
    target_column = train_data.columns[-1]

    y_true_test = test_data[target_column]
    X_test = test_data.iloc[:, :-1]


    for percent in training_percentages:
        number_of_samples =  int(len(train_data) * percent)
        subset_train_data = train_data.sample(n=number_of_samples, random_state=42)

        current_features = list(subset_train_data.columns[:-1]) 

        if subset_train_data.empty:
             print(f"Skipping training size {number_of_samples} (0 samples)")
             continue

        try:
            current_tree = build_tree(subset_train_data, current_features)
            
            
            predictions = []
            for index, row in X_test.iterrows():
                predictions.append(predict(current_tree, row))

           
            correct_predictions = sum(1 for true, pred in zip(y_true_test, predictions) if true == pred)
            accuracy = correct_predictions / len(y_true_test) if len(y_true_test) > 0 else 0

            
            training_sizes.append(number_of_samples)
            test_accuracies.append(accuracy)
            
            print(f"Trained with {number_of_samples} samples, Test Accuracy: {accuracy:.4f}")

        except Exception as e:
            print(f"Error building/predicting with {number_of_samples} samples: {e}")

    if training_sizes:
        ploting.figure(figsize=(10, 6))
        ploting.plot(training_sizes, test_accuracies, marker='o', linestyle='-', label='Test Accuracy')

        ploting.xlabel("Number of Training Examples")
        ploting.ylabel("Accuracy")
        ploting.title("Learning Curve for Decision Tree")
        ploting.grid(True)
        ploting.legend()
        ploting.show()
    else:
        print("No data points collected for learning curve. Plotting skipped.")

# Main excuction block runs all the functions built and prints it out in a nice organised way 
if __name__ == "__main__":
    try:
        
        full_dataset = load_data('car.csv')

        
        train_data, test_data = split_data(full_dataset)
        
        print("--- Data Split ---")
        print(f"Training set size: {len(train_data)}")
        print(f"Test set size:     {len(test_data)}\n")

       
        initial_features = list(train_data.columns[:-1])

      
        print("Building the decision tree on full training data...")
        decision_tree = build_tree(train_data, initial_features)
        print("Tree construction complete!\n")

       
        print("--- Final Tree Evaluation ---")
        true_labels = list(test_data.iloc[:, -1])
        predicted_labels = make_predict(decision_tree, test_data)
        all_class_labels = full_dataset.iloc[:, -1].unique()
        calculate_and_print_metrics(true_labels, predicted_labels, all_class_labels)

       
        plot_learning_curve(train_data, test_data)

    except FileNotFoundError:
        print("Error: 'car.csv' not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")