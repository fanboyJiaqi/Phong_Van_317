import numpy as np


data = np.load('mnist_dataset.npz')
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)


test_indices = np.random.choice(x_test.shape[0], 1000, replace=False)

x_test_subset = x_test[test_indices]
y_test_subset = y_test[test_indices]

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):

        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Lấy chỉ số của k điểm gần nhất
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        

        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)

        max_count_index = np.argmax(counts)
        return unique_labels[max_count_index]


knn = KNN(k=9)
knn.fit(x_train, y_train)


y_pred = knn.predict(x_test_subset)



def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix

# Calculate metrics
def calculate_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    
    # Handle any NaNs that might arise from division by zero
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)

    accuracy = np.sum(tp) / np.sum(cm)

    return accuracy, precision, recall, f1_score, cm, tp, tn, fp, fn

num_classes  =10
accuracy, precision, recall, f1_score, cm, tp, tn, fp, fn = calculate_metrics(y_test, y_pred, num_classes)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Confusion Matrix:\n{cm}")
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
