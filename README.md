# Fake News Detection using Machine Learning

## 📌 Overview
This project is a **Fake News Detection System** built using **Python, NLP, Machine Learning, and Streamlit**. It classifies news articles as either **Fake** or **Real** using a **Naïve Bayes** classifier and **TF-IDF Vectorization**.

## 🛠️ Features
- **Text Preprocessing**: Cleans news text by removing special characters, stopwords, and lemmatizing words.
- **TF-IDF Vectorization**: Converts text into numerical features for machine learning.
- **Naïve Bayes Classifier**: A trained model to predict if news is **Fake** or **Real**.
- **Streamlit Web App**: A user-friendly interface to input news articles and get predictions.
- **Real-Time Fact-Checking**: Cross-verifies with 50+ trusted sources.
- **Image Verification Capabilities**: Helps in verifying images along with news text.
- **Wikipedia Context Lookup**: Fetches related Wikipedia information for better understanding.

---


### 🔍 Verification Tools

#### 🧐 Fact-Check This Claim
Search for: **Pakis**

🔗 [Google Fact-Check Search](https://www.google.com/search?q=Pakis+fact+check)

🔎 "Pakis" hoax  
🔎 "Pakis" debunked  

#### 🌐 Web Verification Tips
- Search trending fact-checks on [Twitter](https://twitter.com/explore)
- Check [Reuters Fact Check](https://www.reuters.com/fact-check/)
- Verify images with [TinEye](https://www.tineye.com/)


## 📂 Dataset
The dataset consists of two CSV files:
- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains real news articles.

The dataset is stored in the **`datasets/`** folder.

---

## 📦 Installation and Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shiva-yadav-ds/Fake_News_Detection_App.git
cd Fake-News-Detection_App
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 5️⃣ Train the Model
Run the **News Classification** script to train the model and save it:
```bash
python news_classification.py
```

This will generate:
- `datasets/vectorizer.pkl` (TF-IDF vectorizer)
- `datasets/fake_news_model.pkl` (Trained model)

### 6️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🔑 API Key Setup
To use real-time fact-checking features, you'll need API keys from different sources. Here’s where to get them:
- **News API** → [Get API Key](https://newsapi.org/)

Ensure your API keys are stored securely and never hardcoded into your repository.

**How to Secure API Keys?**
- Store them in a `.env` or `secrets.toml` file.
- Use environment variables in your code.
- Never push API keys to GitHub.

---

## 🚀 Deploy on GitHub

### 1️⃣ Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit"
```

### 2️⃣ Add Remote Repository
```bash
git remote add origin https://github.com/yourusername/fake-news-detection.git
git branch -M main
git push -u origin main
```

---

## 📸 Screenshots
![image](https://github.com/user-attachments/assets/e614b9c4-5de5-43af-9300-17432561cefd)



![image](https://github.com/user-attachments/assets/aa9cd518-0af5-4d28-a1a2-26d5a7ebb2a1)




---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License
This project is open-source and available under the **MIT License**.

Happy Coding! 🚀
