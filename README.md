# Sentiment Analysis with GloVe Embeddings

A simple text classifier that determines if text is positive, negative, or neutral using pre-trained word embeddings and machine learning.

## How It Works

- **GloVe Embeddings**: Uses pre-trained word vectors instead of training from scratch
- **Document Vectors**: Averages word vectors to represent entire sentences
- **Classification**: Uses Logistic Regression to predict sentiment

## Setup

Install required packages:
```bash
pip install numpy pandas scikit-learn nltk
```

Download GloVe embeddings:
1. Download `glove.6B.zip` from [Stanford GloVe page](https://nlp.stanford.edu/projects/glove/)
2. Extract `glove.6B.100d.txt` to your project folder

## Usage

```bash
python sentiment_analysis_classification.py
```

## Sample Results

**Dataset**: 20 documents (8 positive, 8 negative, 4 neutral)

**Performance**:
- Accuracy: 75%
- Perfect on positive samples
- Good on negative samples  
- Struggled with neutral samples

**Example Predictions**:
```
"This is an absolutely horrible product, I hate it." → negative
"The movie was decent, not bad." → neutral
```

## Key Output

```
Dataset size: 20 documents
Loaded 400,000 word embeddings
Training: 16 samples | Testing: 4 samples
Accuracy: 75%
```

## Requirements

- Python 3.x
- NumPy, Pandas, Scikit-learn, NLTK
- GloVe embeddings file (822 MB download)

## Next Steps

- Use larger datasets
- Try different embedding sizes
- Experiment with neural networks
- Add more text preprocessing