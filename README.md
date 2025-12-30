# Phishing URL & Email Detection

A comprehensive Python project for detecting phishing URLs and emails using machine learning. The system leverages both URL and email feature analysis to provide accurate and interpretable phishing detection.

## Features

- **URL Analysis:** Lexical, structural, and network-based features.
- **Email Analysis:** TF-IDF vectorization of email text combined with URL feature aggregation.
- **Machine Learning:** XGBoost models trained for both URLs and emails.
- **Probability Visualization:** Display prediction probabilities using bar charts.
- **High Accuracy:** URL detection achieves 100% accuracy; email detection achieves 96% accuracy.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/muneebanjum06/ML-Based-Phising-Detector.git 
cd phishing-detector
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Predict a URL

```python
from src.url_feature_extractor import predict_url, predict_with_proba_graph

url = "http://example.com/login"
label, proba = predict_url(url, pipeline, feature_list)
print(label, proba)
predict_with_proba_graph(url, pipeline, feature_list)
```

### Predict an Email

```python
from src.email_feature_extractor import predict_email, plot_probability_bar

email_text = "Dear user, please login at http://fakebank.com..."
label, proba = predict_email(email_text, clf, tfidf, le)
print(label, proba)
plot_probability_bar(email_text, clf, tfidf, le)
```

## Datasets

- `data/PhiUSIIL_Phishing_URL_Dataset.csv`: 235,795 URLs with 77 features.
- `data/phishing_email.csv`: 18,650 email messages labeled as "Phishing Email" or "Safe Email".

## Methodology

1. **URL Feature Extraction:** Length, character distribution, special characters, subdomain analysis, IP-based detection, suspicious keywords.
2. **Email Feature Extraction:** TF-IDF of email text, aggregated URL features, metadata analysis.
3. **Model Training:** XGBoost classifiers with 80-20 stratified train-test split.
4. **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix.

## Results

- **URL Detection:** Perfect classification (100% accuracy) because of the highly seperable dataset, zero false positives/negatives.
- **Email Detection:** High performance (96% accuracy), minor false positives/negatives, interpretable probability outputs.

## Future Work

- Ensemble methods combining URL and email models.
- Deep learning integration (BERT for email, CNN for URL patterns).
- Real-time incremental learning.
- Expanded features (WHOIS data, SSL validation, domain reputation).
- Multi-language support.
- Explainable AI using SHAP or LIME.

## Saved Models

- `models/urlphishing_model.pkl`
- `models/urlfeature_list.pkl`
- `models/phishing_detector_model_xgb.joblib`
- `models/tfidf_vectorizer.joblib`
- `models/label_encoder.joblib`

## References

1. Sahingoz, O. K., et al. (2019). Machine learning based phishing detection from URLs. Expert Systems with Applications, 117, 345–357. [https://doi.org/10.1016/j.eswa.2018.08.031](https://doi.org/10.1016/j.eswa.2018.08.031)
2. Verma, R., & Ranga, V. (2020). Machine learning based intrusion detection systems for IoT applications. Wireless Personal Communications, 111(4), 2287–2310. [https://doi.org/10.1007/s11277-019-06986-8](https://doi.org/10.1007/s11277-019-06986-8)
3. Ma, J., et al. (2009). Learning to detect malicious URLs. ACM Transactions on Intelligent Systems and Technology, 1(1), 1–24. [https://doi.org/10.1145/1858948.1858949](https://doi.org/10.1145/1858948.1858949)
4. Fette, I., et al. (2007). Learning to detect phishing emails. WWW Conference, 649–656. [https://doi.org/10.1145/1242572.1242660](https://doi.org/10.1145/1242572.1242660)
5. Rao, R. S., & Pais, A. R. (2019). Detection of phishing websites using an efficient feature-based machine learning framework. Neural Computing and Applications, 31(8), 3851–3873. [https://doi.org/10.1007/s00521-017-3305-0](https://doi.org/10.1007/s00521-017-3305-0)
6. Scikit-learn: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
7. XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
8. tldextract Python Library: [https://github.com/john-kurkowski/tldextract](https://github.com/john-kurkowski/tldextract)
