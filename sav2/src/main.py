import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from fraud_detection import (
    generate_synthetic_data,
    preprocess_data,
    downsample_data,
    discretize_data,
    build_bayesian_model,
    evaluate_model
)

def plot_results(metrics, y_true, y_pred, y_probs):
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        metrics['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Fraud', 'Fraud'],
        yticklabels=['Non-Fraud', 'Fraud']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Plot ROC curve
    fpr, tpr, _ = metrics['roc_curve']
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data(num_samples=2000, fraud_ratio=0.2)
    
    # Preprocess data
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Downsample data
    print("Downsampling data...")
    df_balanced = downsample_data(df_processed)
    
    # Apply PCA
    print("Applying PCA...")
    X = df_balanced.drop(columns=['fraud_Cases'])
    y = df_balanced['fraud_Cases']
    
    pca = PCA(n_components=7)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame with PCA components
    pca_columns = [f'PCA_{i}' for i in range(1, 8)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)
    df_pca['fraud_Cases'] = y
    
    # Split data
    print("Splitting data...")
    train_df, test_df = train_test_split(
        df_pca,
        test_size=0.2,
        random_state=42,
        stratify=df_pca['fraud_Cases']
    )
    
    # Discretize data
    print("Discretizing data...")
    train_discretized = discretize_data(train_df, pca_columns)
    test_discretized = discretize_data(test_df, pca_columns)
    
    # Build and train model
    print("Building and training Bayesian Network model...")
    model = build_bayesian_model(train_discretized)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_discretized)
    
    # Print results
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Plot results
    plot_results(metrics, test_df['fraud_Cases'].values, metrics['y_pred'], metrics['y_probs'])

if __name__ == "__main__":
    main()