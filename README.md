# GPT-2 Sparse Autoencoder Visualizer

A Streamlit application demonstrating Mechanistic Interpretability by visualizing how a Sparse Autoencoder decomposes GPT-2's internal activations into interpretable features.

## Requirements

- Python 3.10 or higher

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Enter text in the input box
2. Click "Analyze Features"
3. View the top 5 most active SAE features and their activation strengths

## Note

First run will download the GPT-2 model and SAE weights (approximately 500MB). Subsequent runs will be faster due to caching.