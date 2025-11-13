# 🧠 Depression Text Classifier (Naive Bayes)

## Overview  
This project focuses on detecting **depression-related text patterns** using **Natural Language Processing (NLP)** and a **Naive Bayes classifier**.  
The model was trained on Reddit posts to classify whether a text indicates signs of depression or not.  

It combines text preprocessing, feature extraction with TF-IDF, and probabilistic classification to achieve high accuracy in mental health–related content analysis.

---

## Features  
- Text preprocessing: tokenization, stopword removal, and lemmatization using **NLTK** and **spaCy**  
- Feature extraction with **TF-IDF Vectorizer**  
- Classification using **Multinomial Naive Bayes (NB)**  
- Model evaluation using **accuracy**, **precision**, **recall**, and **F1-score**  
- Visualization of frequent words and confusion matrix for performance interpretation  
- Model persistence using **pickle** for reproducibility  

---

## Tech Stack  
- **Python 3.10+**  
- **NLTK** for text processing  
- **spaCy** for lemmatization  
- **scikit-learn** for feature extraction and model training  
- **Matplotlib / Seaborn** for visualization  
- **Pandas / NumPy** for data manipulation  

---

## Project Structure 
    
    ├── dataset/ # dataset (cleaned Reddit depression dataset)
    ├── notebooks/ # Jupyter/Colab notebooks for experimentation
    ├── model/ # saved model (.pkl) and vectorizer
    ├── src/ # Python scripts (if modularized)
    ├── README.md # project documentation
    └── requirements.txt # dependencies

---

## Installation  
    ```bash
    git clone https://github.com/RaffiAkhdilputra/depressed-nb-classifier.git
    cd depressed-nb-classifier
    pip install -r requirements.txt

---

## Usage Example

    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Load trained model and vectorizer
    with open('models/nb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Sample text
    sample_text = ["I feel so lost and empty these days."]

    # Transform and predict
    sample_vector = vectorizer.transform(sample_text)
    prediction = model.predict(sample_vector)
    
    print("Depression Detected" if prediction[0] == 1 else "No Depression Detected")

---

## Results
    
- Achieved ~90% accuracy on test data
- High recall and F1-score for depressive text classification
- Visualizations show strong separation between depressive and non-depressive classes
- Top indicative terms: depression, feel, life, sleep, anxiety

---

## Dataset

Depression: Reddit Dataset (Cleaned)
Source  : https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned

## License

Feel free to use, modify, and distribute it with attribution.

## Contact

Muhammad Raffi Akhdilputra

    Email   : raffiakdilputra123@gmail.com
    GitHub  : @RaffiAkhdilputra
