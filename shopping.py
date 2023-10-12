import csv
import sys
import math
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.4

'''Added my own NNeighbour classifier'''
class NearestNeighborClassifier:
    def __init__(self, k=1):
        self.k = k

    def fit(self, evidence, labels):
        self.evidence = evidence
        self.labels = labels

    def predict(self, test_data):
        predictions = []
        for test_instance in test_data:
            distances = [math.sqrt(sum((x - y) ** 2 for x, y in zip(test_instance, train_instance)) ** 0.5) for train_instance in self.evidence]
            nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
            nearest_labels = [self.labels[i] for i in nearest_indices]
            predictions.append(max(set(nearest_labels), key=nearest_labels.count))
        return predictions


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=TEST_SIZE)

    model=train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity, f1_measure = evaluate(y_test, predictions)

    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"F1 Measure: {f1_measure:.2f}")



def load_data(filename):
   
    evidence, labels = [], []
    import calendar
    month_to_number = {name: num - 1 for num, name in enumerate(calendar.month_abbr) if num}

    with open(filename) as data:
        reader = csv.reader(data)
        next(reader)  # skip the fist line with attributes
        for row in reader:
            evidence.append([
                int(row[0]),    # Administrative
                float(row[1]),  # Administrative_Duration
                int(row[2]),    # Informational
                float(row[3]),  # Informational_Duration
                int(row[4]),    # ProductRelated
                float(row[5]),  # ProductRelated_Duration
                float(row[6]),  # BounceRates
                float(row[7]),  # ExitRates
                float(row[8]),  # PageValues
                float(row[9]),  # SpecialDay
                month_to_number[row[10][:3]],  # Month
                int(row[11]),   # OperatingSystems
                int(row[12]),   # Browser
                int(row[13]),   # Region
                int(row[14]),   # TrafficType
                1 if row[15] == 'Returning_Visitor' else 0,  # VisitorType
                int((row[16]) == 'TRUE'),   # Weekend
            ])

            labels.append(
                int(row[17] == 'TRUE')  # Revenue
            )

    return evidence, labels


def train_model(evidence, labels):
    model = NearestNeighborClassifier(k=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    sensitivity, specificity = 0.0, 0.0
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

    for label, prediction in zip(labels, predictions):
        if label == 1:
            if prediction == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if prediction == 0:
                true_negatives += 1
            else:
                false_positives += 1

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    sensitivity = recall
    specificity = true_negatives / (true_negatives + false_positives)

    if precision + recall == 0:
        f1_measure = 0
    else:
        f1_measure = 2 * (precision * recall) / (precision + recall)

    return sensitivity, specificity, f1_measure



if __name__ == "__main__":
    main()
