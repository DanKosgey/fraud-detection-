import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

from data_processing import (
    generate_synthetic_data, 
    preprocess_data, 
    apply_pca, 
    discretize_data
)
from bayesian_model import predict_probabilities
from visualization import (
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_metrics_comparison
)

def load_model(model_type='bayes'):
    """Load a trained model from file."""
    try:
        import pickle
        model_path = f'models/{model_type}_model.pkl'
        
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found.")
            print("Please run the main pipeline first: python src/main.py")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Successfully loaded {model_type.upper()} model.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_test_data(num_samples=500, fraud_ratio=0.2, n_components=7):
    """Generate and preprocess new test data."""
    print(f"Generating {num_samples} test samples with {fraud_ratio*100}% fraud ratio...")
    
    # Generate new synthetic data
    df = generate_synthetic_data(num_samples, fraud_ratio)
    
    # Preprocess the data
    df_preprocessed = preprocess_data(df)
    
    # Apply PCA
    df_pca = apply_pca(df_preprocessed, n_components)
    
    # Discretize PCA components for Bayesian Network
    pca_columns = [f'PCA_{i+1}' for i in range(n_components)]
    df_discretized = discretize_data(df_pca, pca_columns)
    
    # Split features and target
    X_test = df_discretized.drop(columns=['fraud_Cases'])
    y_test = df_discretized['fraud_Cases']
    
    print(f"Test data prepared with {X_test.shape[1]} features.")
    return X_test, y_test

def evaluate_on_test_data(model, X_test, y_test, model_type='bayes', threshold=0.5):
    """Evaluate model performance on test data."""
    if model is None:
        return None
    
    print(f"Evaluating {model_type.upper()} model on test data...")
    
    # Get probability predictions
    y_probs = predict_probabilities(model, X_test)
    
    # Use a more realistic threshold for Bayesian model
    if model_type.lower() == 'bayes':
        threshold = 0.65  # Higher threshold for better precision
    
    y_pred = (y_probs >= threshold).astype(int)
    
    # Calculate metrics
    y_true = y_test.astype(int)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'y_pred': y_pred,
        'y_probs': y_probs
    }
    
    return metrics

def display_and_save_results(metrics, model_type='bayes'):
    """Display and save evaluation results."""
    if metrics is None:
        return
    
    # Create output directory
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Print metrics
    print("\nTest Results:")
    print("-" * 40)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f"{metric.capitalize()}: {metrics[metric]:.4f}")
    
    # Save metrics to JSON
    metrics_json = {
        k: v.tolist() if hasattr(v, 'tolist') else v
        for k, v in metrics.items()
    }
    
    with open(f'{output_dir}/{model_type}_test_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    # Generate and save plots
    # Confusion matrix
    cm_fig = plot_confusion_matrix(metrics['confusion_matrix'], 
                                  title=f'{model_type.upper()} Model Confusion Matrix (Test Data)')
    cm_fig.savefig(f'{output_dir}/{model_type}_test_confusion_matrix.png')
    
    # ROC curve
    roc_fig = plot_roc_curve(metrics['y_pred'], metrics['y_probs'],
                            title=f'{model_type.upper()} Model ROC Curve (Test Data)')
    roc_fig.savefig(f'{output_dir}/{model_type}_test_roc_curve.png')
    
    # Metrics comparison
    metrics_fig = plot_metrics_comparison(
        {k: metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']},
        title=f'{model_type.upper()} Model Metrics (Test Data)'
    )
    metrics_fig.savefig(f'{output_dir}/{model_type}_test_metrics.png')
    
    plt.close('all')
    print(f"Results and plots saved to {output_dir}/")

def vary_fraud_ratio_test():
    """Test model performance across different fraud ratios."""
    model = load_model(model_type='bayes')
    if model is None:
        return
    
    fraud_ratios = [0.05, 0.1, 0.2, 0.3, 0.4]
    results = {}
    
    print("\nTesting model performance across different fraud ratios...")
    for ratio in fraud_ratios:
        print(f"\nGenerating test data with {ratio*100}% fraud ratio...")
        X_test, y_test = generate_test_data(num_samples=500, fraud_ratio=ratio)
        metrics = evaluate_on_test_data(model, X_test, y_test)
        
        results[ratio] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc']
        }
    
    # Save results
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/fraud_ratio_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics_to_plot:
        values = [results[ratio][metric] for ratio in fraud_ratios]
        plt.plot(fraud_ratios, values, marker='o', label=metric.capitalize())
    
    plt.xlabel('Fraud Ratio')
    plt.ylabel('Metric Value')
    plt.title('Model Performance Across Different Fraud Ratios')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fraud_ratios, [f"{r*100}%" for r in fraud_ratios])
    
    plt.savefig(f'{output_dir}/fraud_ratio_comparison.png')
    plt.close()
    
    print(f"Fraud ratio comparison results saved to {output_dir}/")

def main():
    """Main function to test the model on new data."""
    print("Bank Fraud Detection - Model Testing")
    print("====================================")
    
    # Ensure model directory exists
    if not os.path.exists('models'):
        print("Models directory not found. Please run main.py first to train models.")
        print("Run: python src/main.py")
        return
    
    # First test with default parameters
    model = load_model(model_type='bayes')
    if model:
        X_test, y_test = generate_test_data()
        metrics = evaluate_on_test_data(model, X_test, y_test)
        display_and_save_results(metrics)
        
        # Test model across different fraud ratios
        vary_fraud_ratio_test()
    
    print("\nTesting completed.")

if __name__ == "__main__":
    main() 