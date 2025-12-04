https://ishaanaggrawal-credit-default-predict-app-97qhbv.streamlit.app/

# 💳 Credit Default AI - Risk Assessment Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com) <!-- Replace with your deployed app URL -->

An interactive web application built with Streamlit that uses a machine learning model to predict the probability of credit card payment default.

This tool provides a user-friendly interface for financial risk assessment based on client data, leveraging a powerful XGBoost model.

***

## ✨ Features

- **Interactive UI**: A clean and intuitive web interface powered by Streamlit.
- **Advanced ML Model**: Utilizes an **XGBoost Classifier** for high-accuracy predictions.
- **Dynamic Risk Assessment**: Classifies clients into **Low, Moderate, High,** or **Critical** risk categories based on the predicted default probability.
- **Detailed Input Form**: Captures 23 key features, including demographic data, credit limit, payment history, and bill statement amounts.
- **Visual Feedback**: Presents the final risk assessment in a clear, color-coded result card.

## ⚙️ Tech Stack

- **Backend & ML**: Python
- **Web Framework**: Streamlit
- **Machine Learning**: XGBoost, Scikit-learn (for the scaler)
- **Data Handling**: Pandas, NumPy
- **Model Persistence**: Joblib

## 📊 Dataset

The model was trained on the publicly available [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) from the UCI Machine Learning Repository. This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.8+
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/credit-default-ai.git
    cd credit-default-ai
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file should be created with the following content:
    ```txt
    streamlit
    pandas
    numpy
    xgboost
    scikit-learn
    joblib
    ```
    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

### Model & Scaler Files

Ensure the pre-trained model and scaler files are present in the root directory of the project:
- **Model**: `xgboost_model.json` (or `.ubj`/`.pkl`)
- **Scaler**: `scaler_new.pkl` (or `scaler.pkl`)

If you encounter version mismatch errors with the XGBoost model, you may need to run the provided utility script:
```bash
python fix_models.py
```

### Usage

Once the setup is complete, run the Streamlit application with the following command:

```bash
streamlit run app.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

