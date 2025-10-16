# News-Topic-Classifier-Text-Classification-

Problem: Classify news articles into categories (e.g., comp.graphics, sci.space, rec.autos, etc.).
Dataset: scikit-learn 20 Newsgroups (selected 8 categories for balanced training).
Approach: TF-IDF feature extraction → LinearSVC pipeline.
Modeling: GridSearchCV for hyperparameter tuning (ngram range, C).
Results: ~84% test accuracy; confusion matrix and per-class precision/recall shown in the PDF.
Artifacts: Saved pipeline (news_topic_svm.joblib) and predict() helper with sample predictions.
Key learnings: importance of n-grams and class-balanced sampling, interpretability via top-features per class.
