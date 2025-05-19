# Bank Fraud Detection using Bayesian Networks

This project implements a fraud detection system for bank transactions using Bayesian Networks. The system uses Principal Component Analysis (PCA) for dimensionality reduction and implements both Maximum Likelihood and Bayesian estimation methods for model training.

## Project Structure

```
saviour/
├── data/               # Data storage directory
├── models/            # Trained model storage
├── notebooks/         # Jupyter notebooks for exploration
├── plots/             # Generated visualizations
│   ├── bayes/         # Plots for Bayesian estimation
│   ├── mle/           # Plots for Maximum Likelihood estimation
├── results/           # Model evaluation results
├── src/               # Source code
│   ├── data_processing.py    # Data preparation and preprocessing
│   ├── bayesian_model.py     # Bayesian Network model implementation
│   ├── visualization.py      # Plotting and visualization functions
│   ├── structure_learning.py # Network structure learning
│   └── main.py               # Main execution script
└── tests/             # Unit tests
```

## Key Features

1. **Synthetic Data Generation**: Creates realistic fraud detection data with features that have varying relationships with fraud.
2. **Data Preprocessing Pipeline**: Handles categorical encoding, scaling, and dimensionality reduction.
3. **PCA Transformation**: Reduces dimensionality while preserving variance.
4. **Bayesian Network Modeling**: Implements probabilistic graphical models for fraud classification.
5. **Different Estimation Methods**: Compares Maximum Likelihood (MLE) and Bayesian estimation approaches.
6. **Structure Learning**: Demonstrates the impact of network structure on model performance.
7. **Visualization Suite**: Comprehensive visualization of model results.

## MLE vs Bayesian Estimation

The project demonstrates key differences between Maximum Likelihood Estimation (MLE) and Bayesian estimation:

- **Maximum Likelihood Estimation (MLE)**:
  - Uses frequency counting without prior information
  - Tends to have better recall as it follows the data more closely
  - May overfit to the training data
  - Works well with larger datasets where the data speaks for itself

- **Bayesian Estimation**:
  - Incorporates prior information to regularize parameter estimates
  - Typically has better precision as it's less prone to false positives
  - Better handles sparse or rare events through smoothing
  - More robust against overfitting
  - Can perform better on smaller datasets or with high dimensionality

Our results show that Bayesian estimation typically provides better overall accuracy and precision, while MLE may offer better recall. The choice between these methods depends on the specific application needs - whether false positives or false negatives are more costly.

## Network Structure Comparison

We also compare two approaches to defining the Bayesian Network structure:

1. **Manually defined structure**: Based on domain expertise and assumptions about variable relationships
2. **Learned structure**: Automatically discovered from data using Hill Climbing algorithm

The structure comparison reveals that the learned structure often outperforms the manual structure in our experiments, suggesting that data-driven approaches can uncover non-obvious relationships that human experts might miss.

## Usage

To run the complete pipeline:

```bash
python src/main.py
```

To run with structure comparison:

```bash
python src/main.py --compare-structures
```

## Results

The results are saved in the following directories:
- `results/`: Detailed metrics in JSON format
- `plots/`: Visualizations including ROC curves, confusion matrices, and network structures

## Requirements

See `requirements.txt` for the full list of dependencies. Key libraries include:
- numpy
- pandas
- scikit-learn
- pgmpy
- matplotlib
- seaborn
- networkx

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on research in Bayesian Networks for fraud detection
- Uses the pgmpy library for Bayesian Network implementation
- Implements best practices in machine learning and data preprocessing 