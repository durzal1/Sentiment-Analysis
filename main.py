import torch.nn as nn
import torch.optim as optim
from torchdata.datapipes.iter import IterableWrapper, FileOpener
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from dataset import *



# constants
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = .001
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
embedding_dim = 100
hidden_dim = 100
classes = 1
num_layers = 2
max_words = 200
clip = 5


# data
datapipe = IterableWrapper(["IMDB Dataset.csv"])
datapipe = FileOpener(datapipe, mode='b')
datapipe = datapipe.parse_csv(skip_lines=1)


text_data = []
labels = []
cnt = 0
tokenizer = get_tokenizer('basic_english')
for sample in datapipe:
    cnt += 1

    # adds the data and seperates it
    text_data.append(sample[0])
    labels.append(sample[1])

    # cnt here is used to ensure that it perfectly fits with my batch_size
    if (cnt == 49984):
        break

# Uses the custom dataset I created
dataset = CustomDataset(text_data, labels, max_words)


# Creates a dataloader that we can train with
data_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True)

# How many unique vocab words exist. Very important
num_embeddings = len(dataset.vocab)


# model being used
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(.3)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, hidden):
        embedded = self.embedding(text)
        output, hidden = self.lstm(embedded, hidden)

        # Take the last output from the LSTM (B,S,H) -> (B,H)
        last_output = output[:, -1, :]

        # apply dropout to decrease chance of overfiting
        last_output = self.dropout(last_output)

        # Pass the last output through the fully connected layer
        final = self.fc(last_output)

        out = self.sigmoid(final)

        return out, hidden

    def init_hidden(self):
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.num_layers, BATCH_SIZE, self.hidden_dim)).to(DEVICE)
        c0 = torch.zeros((self.num_layers, BATCH_SIZE, self.hidden_dim)).to(DEVICE)
        hidden = (h0, c0)
        return hidden

# model being used
model = SentimentClassifier(num_embeddings, embedding_dim, hidden_dim, classes,num_layers)
model.to(DEVICE)

# criterion and optimizer for training
criterion = nn.BCELoss() # doesn't add an extra sigmoid since we do that in the forward method
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training
for epoch in range(NUM_EPOCHS):
    # used to see how good it is
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # initialize hidden state
    h = model.init_hidden()

    loop = tqdm(data_loader, leave=True)

    for batch, (x, y) in enumerate(loop):
        # get inputs and targets
        x, y = x.to(DEVICE), y.to(DEVICE)
        h = tuple([each.data for each in h])

        model.zero_grad()

        # forward prop
        out, h = model(x, h)

        # compress and ensure that out and y are both floats for the loss function
        out = out.squeeze()
        out = out.to(torch.float64)
        y = y.to(torch.float64)

        # loss function
        loss = criterion(out, y)


        # back pop for gradients then gradient adjustment
        optimizer.zero_grad()
        loss.backward()

        # help prevent exploding gradient problem
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update gradients
        optimizer.step()

        # accumulate loss
        total_loss += loss.item()

        # calculate accuracy (later on)

        # convert probabilities to binary predictions using a threshold
        pred = torch.round(out.squeeze())
        total_correct += torch.sum(pred == y.squeeze()).item()

        # compute accuracy for the current batch
        total_samples += x.size(0)

        # update progress bar
        loop.set_postfix(loss=loss.item())

    # get statistics
    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples

    if epoch % 1 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
