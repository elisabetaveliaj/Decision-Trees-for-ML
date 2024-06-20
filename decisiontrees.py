import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report, make_scorer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform

# Load data
dataset = pd.read_excel('unbalanced_data.xlsx')
dataset2 = pd.read_excel('data_undersample.xlsx')
dataset3 = pd.read_excel('data_oversample.xlsx')

# Encode target labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['Target'])
X = dataset.drop('Target', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Data preparation - Standardize numerical features
numerical_cols = ["Age at enrollment", "GDP", "Inflation rate", "Unemployment rate", "Admission grade",
                  "Previous qualification (grade)"] + \
                 [col for col in dataset.columns if col.startswith("Curricular")]

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


# Function to calculate additional metrics
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # Calculate metrics
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    neg_pred_value = TN / (TN + FN)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    prevalence = (TP + FN) / (TP + TN + FP + FN)
    detection_rate = TP / (TP + TN + FP + FN)
    detection_prevalence = (TP + FP) / (TP + TN + FP + FN)
    balanced_accuracy = (sensitivity + specificity) / 2

    metrics = {
        'Sensitivity (Recall)': sensitivity,
        'Specificity': specificity,
        'Pos Pred Value (Precision)': precision,
        'Neg Pred Value': neg_pred_value,
        'F1 Score': f1,
        'Prevalence': prevalence,
        'Detection Rate': detection_rate,
        'Detection Prevalence': detection_prevalence,
        'Balanced Accuracy': balanced_accuracy
    }

    return metrics


# Cross-validation setup
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Cross-validation
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    kappa_scorer = make_scorer(cohen_kappa_score)
    kappa_scores = cross_val_score(model, X_train, y_train, cv=10, scoring=kappa_scorer)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    metrics = calculate_metrics(y_test, y_pred)
    metrics['Accuracy'] = accuracy
    metrics['Cohen\'s Kappa'] = kappa

    return metrics, accuracy_scores.mean(), accuracy_scores.std(), kappa_scores.mean(), kappa_scores.std()


def plot_decision_tree(model, feature_names, title):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True, class_names=label_encoder.classes_)
    plt.title(title)
    plt.show()


def plot_top_feature_importance(model, feature_names, title, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(top_n), top_importances, align="center")
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.ylim([-1, top_n])
    plt.show()


# 1. Single decision tree without any filtering
tree_no_filter = DecisionTreeClassifier(random_state=42)
metrics_no_filter, acc_mean_no_filter, acc_std_no_filter, kappa_mean_no_filter, kappa_std_no_filter = evaluate_model(
    tree_no_filter, X_train, y_train, X_test, y_test)
plot_decision_tree(tree_no_filter, X.columns, "Decision Tree without Filtering")
plot_top_feature_importance(tree_no_filter, X.columns, "Feature Importance without Filtering")

# 2. Single decision tree with pruning
dt_model = DecisionTreeClassifier(random_state=42)
params = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'ccp_alpha': uniform(0.0, 0.1)
}
dt_random_search = RandomizedSearchCV(dt_model, param_distributions=params, n_iter=100, cv=10, n_jobs=-1,
                                      random_state=42)
dt_random_search.fit(X_train, y_train)
best_params = dt_random_search.best_params_
pruned_tree = DecisionTreeClassifier(random_state=42, **best_params)
metrics_pruned_tree, acc_mean_pruned, acc_std_pruned, kappa_mean_pruned, kappa_std_pruned = evaluate_model(
    pruned_tree, X_train, y_train, X_test, y_test)
plot_decision_tree(pruned_tree, X.columns, "Pruned Decision Tree")
plot_top_feature_importance(pruned_tree, X.columns, "Feature Importance of Pruned Tree")

# 3. Single decision tree with bagging
bagging_tree = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42, ccp_alpha=0),
                                 n_estimators=100, random_state=42, oob_score=True)
metrics_bagging_tree, acc_mean_bagging, acc_std_bagging, kappa_mean_bagging, kappa_std_bagging = evaluate_model(
    bagging_tree, X_train, y_train, X_test, y_test)
oob_score_bagging = bagging_tree.oob_score_
# visualize the first tree.
plot_decision_tree(bagging_tree.estimators_[0], X.columns, "Bagging Decision Tree (First Tree)")
plot_top_feature_importance(bagging_tree.estimators_[0], X.columns, "Feature Importance of Bagging (First Tree)")

# 4. Single decision tree with pruning and bagging
bagging_pruned_tree = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42, **best_params),
                                        n_estimators=100, random_state=42, oob_score=True)
metrics_bagging_pruned_tree, acc_mean_bagging_pruned, acc_std_bagging_pruned, kappa_mean_bagging_pruned, kappa_std_bagging_pruned = evaluate_model(
    bagging_pruned_tree, X_train, y_train, X_test, y_test)
oob_score_bagging_pruned = bagging_pruned_tree.oob_score_
# Visualize the first tree.
plot_decision_tree(bagging_pruned_tree.estimators_[0], X.columns, "Bagging Pruned Decision Tree (First Tree)")
plot_top_feature_importance(bagging_pruned_tree.estimators_[0], X.columns,
                            "Feature Importance of Bagging Pruned (First Tree)")

# Save results to an Excel file
results = {
    'Metrics': list(metrics_no_filter.keys()),
    'No Filtering': list(metrics_no_filter.values()),
    'Pruned': list(metrics_pruned_tree.values()),
    'Bagging': list(metrics_bagging_tree.values()),
    'Pruned and Bagging': list(metrics_bagging_pruned_tree.values())
}

cross_val_results = {
    'Model': ['No Filtering', 'Pruned', 'Bagging', 'Pruned and Bagging'],
    'Accuracy Mean': [acc_mean_no_filter, acc_mean_pruned, acc_mean_bagging, acc_mean_bagging_pruned],
    'Accuracy Std': [acc_std_no_filter, acc_std_pruned, acc_std_bagging, acc_std_bagging_pruned],
    'Kappa Mean': [kappa_mean_no_filter, kappa_mean_pruned, kappa_mean_bagging, kappa_mean_bagging_pruned],
    'Kappa Std': [kappa_std_no_filter, kappa_std_pruned, kappa_std_bagging, kappa_std_bagging_pruned]
}

results_df = pd.DataFrame(results)
cross_val_results_df = pd.DataFrame(cross_val_results)

# Adding a dummy sheet
with pd.ExcelWriter('model_metrics.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Metrics', index=False)
    cross_val_results_df.to_excel(writer, sheet_name='Cross-Validation', index=False)
    # Adding a dummy sheet
    workbook = writer.book
    workbook.create_sheet('Dummy')
    if 'Dummy' in writer.sheets:
        workbook.remove(workbook['Dummy'])

# Print the OOB scores
print("OOB score for Bagging Tree:", oob_score_bagging)
print("OOB score for Bagging Pruned Tree:", oob_score_bagging_pruned)
print(best_params)

