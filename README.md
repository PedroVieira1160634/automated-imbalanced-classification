# Automated Imbalanced Classification
A python project that automates imbalanced classification.

This project is composed of two modules:
- *learning module*
- *recommendation module*

In the *learning module* the goal is to combine several resampling and classification algorithms, select the best combination of both to handle a dataset and save the dataset meta-features, the evaluation metrics and the execution time of the best combination into the knowledge base.

In the *recommendation module*, with the assistance of the previous knowledge base constructed, it can recommend the best combination of
resampling and classification algorithms to handle a new imbalanced dataset imported. This recommendation is made by finding in the knowledge base the most similar dataset in terms of meta-features.
