import pandas as pd
import numpy as np
import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


## ------------ PREPROCESS A SINGLE STRING OF TEXT (Adapted from previous function) --------------------------------------

def preprocess_document_text(text):

    if not isinstance(text, str) or not text.strip():
        return []

    text = text.lower()
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))       # creates a translation table that maps string.punctuation (.,!?) to a space character

    text = text.translate(translator)
    text = re.sub(r'\s+', ' ', text).strip()            # All multiple spaces are eliminated and strip removes leading and trailing spaces

    if not text:
        return []

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return lemmatized_tokens


## ------------ LOAD GLOVE EMBEDDINGS (Re-using existing function) --------------------------------------
def load_glove_embeddings(glove_file_path):
    
    word_embeddings = {}
    embedding_dim = 0
    try:
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                if embedding_dim == 0:
                    embedding_dim = len(vector)
                word_embeddings[word] = vector
        print(f"Loaded {len(word_embeddings)} word embeddings from {glove_file_path}")
        print(f"Embedding dimension: {embedding_dim}")
    except FileNotFoundError:
        print(f"Error: GloVe file not found at {glove_file_path}. Please check the path and download.")
        return {}, 0
    except Exception as e:
        print(f"Error loading GloVe embeddings: {e}")
        return {}, 0
    return word_embeddings, embedding_dim

## ------------ GET AVERAGE EMBEDDING (Re-using existing function) --------------------------------------

def get_average_embedding(tokens, embeddings_dict, embedding_dim):

    vector_sum = np.zeros(embedding_dim, dtype='float32')
    count = 0
    for token in tokens:
        if token in embeddings_dict:
            vector_sum += embeddings_dict[token]
            count += 1
    if count > 0:
        return vector_sum / count
    else:
        return np.zeros(embedding_dim, dtype='float32')


## -----------------------------------------------------------------------------
## MAIN EXECUTION FOR CLASSIFICATION TASK
## -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    glove_file_path = "glove-6B-100d.txt" # IMPORTANT: Update this path to your downloaded GloVe file

    # --- 1. Create a small text dataset ---

    data = [
        ("This is a fantastic movie! I loved every minute.", "positive"),
        ("The service was terrible and the food was cold.", "negative"),
        ("What a great experience, highly recommend.", "positive"),
        ("I'm so disappointed with this product.", "negative"),
        ("Absolutely brilliant performance.", "positive"),
        ("It was okay, nothing special.", "neutral"),
        ("Worst purchase ever, completely useless.", "negative"),
        ("Enjoyed the show, very entertaining.", "positive"),
        ("Could have been better, quite boring.", "negative"),
        ("A truly wonderful day.", "positive"),
        ("This is just awful.", "negative"),
        ("Very good quality for the price.", "positive"),
        ("Not happy with the delivery time.", "negative"),
        ("Excellent customer support.", "positive"),
        ("Completely satisfied.", "positive"),
        ("I regret buying this.", "negative"),
        ("Neutral feelings about it.", "neutral"),
        ("Simply amazing!", "positive"),
        ("A total waste of money.", "negative"),
        ("It works as expected.", "neutral")
    ]

    df = pd.DataFrame(data, columns=['text', 'label'])
   
    print(df.head())
    print(f"\nDataset size: {len(df)} documents")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # --- 2. Load pre-trained word embeddings ---
    word_embeddings, embedding_dim = load_glove_embeddings(glove_file_path)

    if embedding_dim == 0:
        print("Error: Could not load GloVe embeddings. Exiting.")
        exit()

    # --- 3. Represent documents by averaging pre-trained word embeddings ---
    print("\n--- Preprocessing and Vectorizing Documents ---")
    df['preprocessed_text'] = df['text'].apply(preprocess_document_text)
    df['document_embedding'] = df['preprocessed_text'].apply(
        lambda tokens: get_average_embedding(tokens, word_embeddings, embedding_dim)
    )

    # Convert list of arrays to a 2D NumPy array for scikit-learn
    X = np.array(df['document_embedding'].tolist())
    y = df['label'].values # Labels

    print(f"\nShape of document embeddings (X): {X.shape}")
    print(f"Shape of labels (y): {y.shape}")

    # --- 4. Split data into training and testing sets ---
    # Stratify by 'y' to maintain label distribution in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # --- 5. Train a simple sklearn.linear_model.LogisticRegression ---
       
    # solver='liblinear' is good for small datasets and binary/multiclass classification
    model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)

    model.fit(X_train, y_train)
   

    # --- 6. Evaluate the model ---
    
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score (weighted): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n--- Test with a new unseen sentence ---")
    new_sentence = "This is an absolutely horrible product, I hate it."
    preprocessed_new_sentence = preprocess_document_text(new_sentence)
    new_sentence_embedding = get_average_embedding(preprocessed_new_sentence, word_embeddings, embedding_dim)

    # Reshape for prediction (model expects 2D array)
    new_sentence_embedding = new_sentence_embedding.reshape(1, -1)

    predicted_label = model.predict(new_sentence_embedding)[0]
    print(f"New Sentence: \"{new_sentence}\"")
    print(f"Predicted Label: {predicted_label}")

    new_sentence_2 = "The movie was decent, not bad."
    preprocessed_new_sentence_2 = preprocess_document_text(new_sentence_2)
    new_sentence_embedding_2 = get_average_embedding(preprocessed_new_sentence_2, word_embeddings, embedding_dim)
    new_sentence_embedding_2 = new_sentence_embedding_2.reshape(1, -1)
    predicted_label_2 = model.predict(new_sentence_embedding_2)[0]
    print(f"New Sentence: \"{new_sentence_2}\"")
    print(f"Predicted Label: {predicted_label_2}")