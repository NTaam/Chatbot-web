import torch
import torch.nn as nn
import torch.nn.functional as F

# class IntentClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, embedding_dim, num_classes, dropout_prob=0.5):
#         super(IntentClassifier, self).__init__()
#         self.embedding = nn.Embedding(input_size, embedding_dim)
#         # BiLSTM: bidirectional=True để sử dụng cả 2 hướng
#         self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
#         self.dropout = nn.Dropout(dropout_prob)  # Lớp Dropout để giảm overfitting
#         self.fc = nn.Linear(hidden_size * 2, num_classes)  # Nhân đôi hidden_size vì BiLSTM có 2 hướng
 
#     def forward(self, x):
#         # x: [batch_size, seq_length]
#         embedded = self.embedding(x)      # [batch_size, seq_length, embedding_dim]
#         _, (hidden, _) = self.rnn(embedded)
#         # hidden[-2:] chứa cả 2 hướng (forward và backward)
#         hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch_size, hidden_size * 2]
#         hidden = self.dropout(hidden)  # Áp dụng Dropout
#         output = self.fc(hidden)           # [batch_size, num_classes]
#         return output




class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_classes, dropout_prob=0.5):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)  # Lớp Dropout để giảm overfitting
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Nhân đôi hidden_size vì BiLSTM có 2 hướng

    def forward(self, x):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)      # [batch_size, seq_length, embedding_dim]
        _, (hidden, _) = self.rnn(embedded)
        # hidden[-2:] chứa cả 2 hướng (forward và backward)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch_size, hidden_size * 2]
        hidden = self.dropout(hidden)  # Áp dụng Dropout
        output = self.fc(hidden)           # [batch_size, num_classes]
        return output