import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

def generate_synthetic_data(num_samples=2000, fraud_ratio=0.2):
    np.random.seed(42)
    n_fraud = int(num_samples * fraud_ratio)
    n_nonfraud = num_samples - n_fraud

    data = {
        'income': np.random.uniform(0.1, 0.9, num_samples),
        'name_email_similarity': np.random.uniform(0, 1, num_samples),
        'prev_address_months_count': np.random.choice([-1] + list(range(0, 381)), num_samples),
        'current_address_months_count': np.random.choice([-1] + list(range(0, 430)), num_samples),
        'customer_age': np.random.randint(10, 91, num_samples),
        'days_since_request': np.random.randint(0, 80, num_samples),
        'intended_balcon_amount': np.random.uniform(-16, 114, num_samples),
        'payment_type': np.random.choice(['A', 'B', 'C', 'D', 'E'], num_samples),
        'zip_count_4w': np.random.randint(1, 6831, num_samples),
        'velocity_6h': np.random.uniform(-175, 16819, num_samples),
        'velocity_24h': np.random.randint(1297, 9587, num_samples),
        'velocity_4w': np.random.randint(2825, 7021, num_samples),
        'bank_branch_count_8w': np.random.randint(0, 2405, num_samples),
        'date_of_birth_distinct_emails_4w': np.random.randint(0, 40, num_samples),
        'employment_status': np.random.choice(['E1', 'E2', 'E3', 'E4', 'E5'], num_samples),
        'credit_risk_score': np.random.uniform(-191, 390, num_samples),
        'email_is_free': np.random.choice([0, 1], num_samples),
        'housing_status': np.random.choice(['H1', 'H2', 'H3'], num_samples),
        'phone_home_valid': np.random.choice([0, 1], num_samples),
        'phone_mobile_valid': np.random.choice([0, 1], num_samples),
        'bank_months_count': np.random.choice([-1] + list(range(0, 33)), num_samples),
        'has_other_cards': np.random.choice([0, 1], num_samples),
        'proposed_credit_limit': np.random.randint(200, 2001, num_samples),
        'foreign_request': np.random.choice([0, 1], num_samples),
        'source': np.random.choice(['INTERNET', 'TELEAPP'], num_samples),
        'session_length_in_minutes': np.random.choice([-1] + list(range(0, 108)), num_samples),
        'device_os': np.random.choice(['Windows', 'macOS', 'Linux'], num_samples),
        'keep_alive_session': np.random.choice([0, 1], num_samples),
        'device_distinct_emails': np.random.choice([-1, 0, 1], num_samples),
        'device_fraud_count': np.random.choice([0, 1], num_samples),
        'month': np.random.randint(0, 12, num_samples),
        'fraud_Cases': [1] * n_fraud + [0] * n_nonfraud
    }

    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def preprocess_data(df):
    categorical_features = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
    numerical_features = [col for col in df.columns if col not in categorical_features + ['fraud_Cases']]

    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cats = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))

    df = df.drop(columns=categorical_features).reset_index(drop=True)
    return pd.concat([df, encoded_df], axis=1).select_dtypes(include=[np.number])

def downsample_data(df, target='fraud_Cases'):
    fraud = df[df[target] == 1]
    non_fraud = df[df[target] == 0]
    non_fraud_downsampled = non_fraud.sample(n=len(fraud) * 4, random_state=42)
    return pd.concat([fraud, non_fraud_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

def discretize_data(data, pca_columns, bins=3):
    discretized = data.copy()
    for col in pca_columns:
        discretized[col] = pd.qcut(
            pd.to_numeric(discretized[col], errors='coerce').fillna(discretized[col].median()),
            q=bins,
            labels=['low', 'med', 'high'],
            duplicates='drop'
        ).astype(str)
    return discretized

def build_bayesian_model(train_data):
    structure = [(f'PCA_{i}', 'fraud_Cases') for i in range(1, 8)]
    
    model = DiscreteBayesianNetwork(structure)
    model.fit(
        train_data,
        estimator=BayesianEstimator,
        prior_type='BDeu',
        equivalent_sample_size=10
    )
    return model

def evaluate_model(model, test_data):
    infer = VariableElimination(model)
    pred_probs = []

    for _, row in test_data.drop(columns=['fraud_Cases']).iterrows():
        evidence = {var: val if val in model.states[var] else list(model.states[var])[0]
                   for var, val in row.to_dict().items()}
        try:
            prob = infer.query(variables=['fraud_Cases'], evidence=evidence).values[1]
            pred_probs.append(prob)
        except:
            pred_probs.append(0.5)

    y_true = test_data['fraud_Cases'].astype(int).values
    y_pred = (np.array(pred_probs) >= 0.5).astype(int)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, pred_probs),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'y_pred': y_pred,
        'y_probs': pred_probs
    }