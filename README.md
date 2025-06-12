# Fake-News-Detection-Using-Text-Analysis
# 📰 Fake News Detection Using Text Analysis

A machine learning project that detects fake news articles using Natural Language Processing (NLP) and classification algorithms. This system helps identify misinformation by analyzing the text content of news articles.

![Fake News Illustration](https://cdn.pixabay.com/photo/2020/04/26/20/03/fake-news-5090739_1280.jpg)

---

## 📚 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---

## 🔎 About the Project

Fake news is a serious problem that can influence public opinion and harm society. This project uses NLP techniques and machine learning models to classify news articles as **real** or **fake** based on their text content.

---

## 📊 Dataset

- **Dataset Name:** Fake and Real News Dataset
- **Size:** ~40,000 articles
- **Classes:** `REAL`, `FAKE`
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 🛠 Technologies Used

- Python 3.8+
- Scikit-learn
- NLTK / spaCy
- Pandas / NumPy
- Matplotlib / Seaborn
- Flask / Streamlit (optional for web deployment)

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and place the dataset:**
   - Place the CSV file (`True.csv`, `Fake.csv`, or merged `news.csv`) into the `/data` directory.

---

## 🚀 Usage

### Train the Model
```bash
python train.py
```

### Evaluate the Model
```bash
python evaluate.py
```

### Predict New Text
```bash
python predict.py --text "The news article text goes here"
```

### Launch Web App (Optional)
```bash
streamlit run app.py
```

---

## 🧠 Model Details

- **Text Preprocessing:**
  - Lowercasing
  - Removing punctuation and stopwords
  - Lemmatization
  - TF-IDF vectorization

- **Classifiers Tested:**
  - Logistic Regression ✅
  - Naive Bayes
  - Random Forest
  - Support Vector Machine (SVM)

- **Best Performing Model:** Logistic Regression
- **Accuracy:** ~96% on test set

---

## 📈 Results

| Metric     | Value    |
|------------|----------|
| Accuracy   | 96.2%    |
| Precision  | 96.4%    |
| Recall     | 95.9%    |
| F1 Score   | 96.1%    |

Confusion matrix, classification report, and ROC curve plots can be found in the `results/` folder.

---

## 🔮 Future Improvements

- Deploy a real-time fake news checker with Flask/Streamlit
- Include headline-only classification
- Expand dataset for multilingual support
- Incorporate deep learning (BERT, LSTM)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## ✉️ Contact

**Your Name**  
📧 your.email@example.com  
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile)

---

## 🙏 Acknowledgements

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [NLTK](https://www.nltk.org/)
