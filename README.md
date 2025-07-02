<div align="center"><img src="SpamlyserLogo.png" style="width: 220px; height: 220px;" /></div>

# <div align="center">SPAMLYSER</div>

## 🛡️ Advanced SMS Spam Detection & Analysis with Transformers

**Spamlyser Pro** is a powerful, real-time SMS spam detection platform built with **Streamlit** and **Transformers**, backed by 4 custom-trained LLM backbones **DistilBERT**, **BERT**, **RoBERTa**, and **ALBERT**. It allows users to classify SMS messages as *SPAM* or *HAM*, visualise spam distribution, and analyse risky message features with an elegant, responsive UI.

---

## ✨ Features

### 🤖 Transformer-Based Classification

* Compare performance across 4 LLM backbones (DistilBERT, BERT, RoBERTa, ALBERT)
* Trained on [HuggingFace’s `sms_spam`](https://huggingface.co/datasets/sms_spam) dataset
* Real-time predictions with confidence scores

### 🔍 Message Feature Analysis

* Length, word count, digit and symbol ratio
* URL and phone number detection
* Uppercase and punctuation overuse analysis

### ⚠️ Risk Indicators

* Flags spam indicators (e.g., URLs, ALL CAPS, exclamations, spam keywords)
* Visual threat insights for each message

### 📊 Live Performance Metrics

* Session-based spam/ham distribution pie charts
* Model-wise classification count tracking

### 🎯 Clean, Professional UI

* Responsive dark mode interface
* Custom model selection sidebar
* Stylish prediction cards with emoji & confidence

---

## 🚀 Live Demo

Experience Spamlyser live here: 
👉 [![**Spamlyser**](https://img.shields.io/badge/View-Live%20Demo-brightgreen?style=for-the-badge)](https://www.google.com/search?q=https://Spamlyser.streamlit.app)

 <div align="center">
 <p>

[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat)
![Visitors](https://api.visitorbadge.io/api/Visitors?path=eccentriccoder01%2FSpamlyser%20&countColor=%23263759&style=flat)
![GitHub Forks](https://img.shields.io/github/forks/eccentriccoder01/Spamlyser)
![GitHub Repo Stars](https://img.shields.io/github/stars/eccentriccoder01/Spamlyser)
![GitHub Contributors](https://img.shields.io/github/contributors/eccentriccoder01/Spamlyser)
![GitHub Last Commit](https://img.shields.io/github/last-commit/eccentriccoder01/Spamlyser)
![GitHub Repo Size](https://img.shields.io/github/repo-size/eccentriccoder01/Spamlyser)
![GitHub Total Lines](https://sloc.xyz/github/eccentriccoder01/Spamlyser)
![Github](https://img.shields.io/github/license/eccentriccoder01/Spamlyser)
![GitHub Issues](https://img.shields.io/github/issues/eccentriccoder01/Spamlyser)
![GitHub Closed Issues](https://img.shields.io/github/issues-closed-raw/eccentriccoder01/Spamlyser)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/eccentriccoder01/Spamlyser)
![GitHub Closed Pull Requests](https://img.shields.io/github/issues-pr-closed/eccentriccoder01/Spamlyser)
 </p>
 </div>

## 📸 Screenshots

<div align="center"><img src="App.jpeg"/></div>
<div align="center"><img src="Results.jpeg"/></div>

---

## 🧠 Models Used

| Model      | Repo URL                                                                        | Characteristics     |
| ---------- | ------------------------------------------------------------------------------- | ------------------- |
| DistilBERT | [🔗 Link](https://huggingface.co/mreccentric/distilbert-base-uncased-spamlyser) | Lightweight & Fast  |
| BERT       | [🔗 Link](https://huggingface.co/mreccentric/bert-base-uncased-spamlyser)       | Balanced & Standard |
| RoBERTa    | [🔗 Link](https://huggingface.co/mreccentric/roberta-base-spamlyser)            | Robust & Accurate   |
| ALBERT     | [🔗 Link](https://huggingface.co/mreccentric/albert-base-v2-spamlyser)          | Efficient & Compact |

---

## 📺 Video Explanation

For a detailed walkthrough of Spamlyser's features and how to use them, check out this video:

**[Insert YouTube Video Link Here]**

---

## 🛠️ Technologies Used

| Tool/Library         | Purpose                             |
| -------------------- | ----------------------------------- |
| **Python**           | Core backend                        |
| **Streamlit**        | Web app interface                   |
| **Transformers**     | Model loading and inference         |
| **Hugging Face Hub** | Model hosting & deployment          |
| **Pandas & Plotly**  | Data processing & visualization     |
| **Regex, Pathlib**   | Feature engineering & file handling |

---

## ⚙️ Installation and Setup

> Clone and run locally using Python and Streamlit.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/eccentriccoder01/Spamlyser.git
   cd Spamlyser
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

---

## Issue Creation ✴
Report bugs and  issues or propose improvements through our GitHub repository.

## Contribution Guidelines 📑

- Firstly Star(⭐) the Repository
- Fork the Repository and create a new branch for any updates/changes/issue you are working on.
- Start Coding and do changes.
- Commit your changes
- Create a Pull Request which will be reviewed and suggestions would be added to improve it.
- Add Screenshots and updated website links to help us understand what changes is all about.

- Check the [CONTRIBUTING.md](CONTRIBUTING.md) for detailed steps...

    
## Contributing is fun🧡

We welcome all contributions and suggestions!
Whether it's a new feature, design improvement, or a bug fix — your voice matters 💜

Your insights are invaluable to us. Reach out to us team for any inquiries, feedback, or concerns.

## 📄 License

This project is open-source and available under the MIT License.

## 📞 Contact

Developed by [Eccentric Explorer](https://eccentriccoder01.github.io/Me)

Feel free to reach out with any questions or feedback\!