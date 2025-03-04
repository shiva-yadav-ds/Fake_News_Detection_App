# Fake News Detection using Machine Learning

## 📌 Overview
This project is a **Fake News Detection System** built using **Python, NLP, Machine Learning, and Streamlit**. It classifies news articles as either **Fake** or **Real** using a **Naïve Bayes** classifier and **TF-IDF Vectorization**.

## 🛠️ Features
- **Text Preprocessing**: Cleans news text by removing special characters, stopwords, and lemmatizing words.
- **TF-IDF Vectorization**: Converts text into numerical features for machine learning.
- **Naïve Bayes Classifier**: A trained model to predict if news is **Fake** or **Real**.
- **Streamlit Web App**: A user-friendly interface to input news articles and get predictions.

---

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
![Fake News Detection UI](https://via.placeholder.com/800x400.png?text=Fake+News+Detection+UI)
![image](https://github.com/user-attachments/assets/85b61a4d-d355-4f88-b83c-7cc25f45c55b)


---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License
This project is open-source and available under the **MIT License**.

Happy Coding! 🚀
