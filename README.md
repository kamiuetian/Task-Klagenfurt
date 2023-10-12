# Task-Klagenfurt

**Nearest Neighbor Classifier for Shopping Data**

**Description:**

This Python code provides a simple implementation of a Nearest Neighbor Classifier for analyzing shopping data. The classifier predicts whether a visitor will make a purchase based on various features. The code includes data loading, training a custom classifier, and evaluating the model's performance using sensitivity, specificity, and the F1 measure.

**Code Components:**

1. **NearestNeighborClassifier:**

   - This class defines a custom Nearest Neighbor Classifier.
   - The constructor allows you to set the number of nearest neighbors (`k`) for the classifier. By default, `k` is set to 1.
   - The `fit` method is used to train the classifier with training data, which includes evidence (features) and corresponding labels (0 for no purchase, 1 for purchase).
   - The `predict` method makes predictions for test data instances based on the k-nearest neighbors from the training data.

2. **main():**

   - The `main` function is the entry point of the code.
   - It loads shopping data from a CSV file, splits the data into training and testing sets, and trains a Nearest Neighbor Classifier using the training data.
   - The trained model is used to predict outcomes for the test data, and the sensitivity, specificity, and F1 measure are calculated to evaluate the model's performance.
   - The results are printed to the console.

3. **load_data(filename):**

   - The `load_data` function reads shopping data from a CSV file.
   - It extracts features (evidence) and labels (purchase or no purchase) from the dataset.
   - Categorical data, such as the month and visitor type, is preprocessed for model compatibility.

4. **train_model(evidence, labels):**

   - The `train_model` function initializes and trains a Nearest Neighbor Classifier using the provided evidence (features) and labels.

5. **evaluate(labels, predictions):**
   - The `evaluate` function calculates sensitivity (true positive rate), specificity (true negative rate), and the F1 measure to assess the classifier's performance.

**Usage:**
python shopping.py shopping.csv

1. To run the code, provide the path to the shopping data CSV file as a command-line argument.
2. The code will split the data, train the model, make predictions, and print evaluation metrics.
