import json
import torch
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import preprocess_text, tokenize
from train import predict_intent, label_to_intent  # Reuses your trained model’s predict_intent
from model import IntentClassifierBiLSTM
from preprocessing import load_data_and_build_vocab

# -----------------------------------------------------------------------------
# 1) Load the data and model
# -----------------------------------------------------------------------------
def load_model_and_data(json_path="data.json", model_path="Nino.pth", vocab_path="vocab.pkl"):
    """
    Loads:
      - your dataset from data.json
      - the trained model from Nino.pth
      - the vocab from vocab.pkl
    Returns the model, vocab, data, label_to_intent, etc.
    """

    # 1. Load the raw data
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 2. Build vocab, get label mappings
    data, vocab, intent_labels, l2i, train_data = load_data_and_build_vocab(json_path)

    # 3. Load model
    input_size = len(vocab)
    hidden_size = 128
    embedding_dim = 64
    num_classes = len(intent_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IntentClassifierBiLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, vocab, raw_data, l2i, intent_labels


# -----------------------------------------------------------------------------
# 2) Evaluate function
# -----------------------------------------------------------------------------
def evaluate():
    # Load everything
    model, vocab, raw_data, l2i, intent_labels = load_model_and_data()

    all_labels = []
    all_preds = []

    # We’ll go through every pattern in data.json and get a prediction
    for intent_dict in raw_data["intents"]:
        true_tag = intent_dict["tag"]
        true_label = intent_labels[true_tag]


        for pattern in intent_dict["patterns"]:
            # Predict the intent
            predicted_tag = predict_intent(pattern)  # Reuses your function from train.py
            predicted_label = intent_labels[true_tag]

            all_labels.append(true_label)
            all_preds.append(predicted_label)

    # Compute metrics
    report = classification_report(all_labels, all_preds, target_names=[k for k in intent_labels.keys()])
    cm = confusion_matrix(all_labels, all_preds)

    print("========== Classification Report ==========")
    print(report)
    print("========= Confusion Matrix (label indices) =========")
    print(cm)

    """
    'cm' is a 2D array where:
      cm[i][j] = number of samples whose true label is i but predicted label is j
    """


# -----------------------------------------------------------------------------
# 3) Main entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate()
