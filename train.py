import argparse
import os
import nltk
import nltk.tokenize
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle


class TextGenModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(TextGenModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, batch_first=True,
                          bidirectional=False)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))


script = os.path(__file__)

parser = argparse.ArgumentParser(prog=script.name)
parser.add_argument('-ind', '--input_dir', required=False, help='Path to data')
parser.add_argument('-m', '--model', required=True, help='Path to model')

args = parser.parse_args()
if args.input_dir:
    data_path = args.input_dir
else:
    data_path = input('Введите путь к данным:')
print(data_path)

model_path = args.model
print(model_path)

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU')
else:
    print('Training on CPU')

text_sample = ''
for file in os.listdir(data_path):
    with open(data_path + '/' + file, encoding='utf-8') as text_file:
        text_from_file = text_file.readlines()
    text_sample += ' '.join(text_from_file)

nltk.download("punkt", quiet=True)
tokens = []
for token in nltk.tokenize.sent_tokenize(text_sample):
    for t in list(map(str.lower, nltk.tokenize.word_tokenize(token))):
        if t.isalnum():
            tokens.append(t)

trigrams = [([tokens[i], tokens[i + 1]], tokens[i + 2])
            for i in range(len(tokens) - 2)]
chunk_len = len(trigrams)
print(trigrams[:3])

vocab = set(tokens)
voc_len = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}

trains = []
targets = []
for context, target in trigrams:
    context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
    trains.append(context_idxs)
    targ = torch.tensor([word_to_ix[target]], dtype=torch.long)
    targets.append(targ)

n_epochs = 10
hidden_size = 100
n_layers = 5
lr = 0.01

model = TextGenModel(voc_len, hidden_size, voc_len, n_layers)
model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train(inp, target):
    if train_on_gpu:
        hidden = model.init_hidden().cuda()
    else:
        hidden = model.init_hidden()
    model.zero_grad()
    loss = 0
    if train_on_gpu:
        for c in range(chunk_len):
            output, hidden = model(inp[c].cuda(), hidden)
            loss += criterion(output, target[c].cuda())
    else:
        for c in range(chunk_len):
            output, hidden = model(inp[c], hidden)
            loss += criterion(output, target[c])

    loss.backward()
    model_optimizer.step()

    return loss.data.item() / chunk_len


all_losses = []
loss_avg = 0
if train_on_gpu:
    model.cuda()
for epoch in range(1, n_epochs + 1):
    loss = train(trains, targets)
    loss_avg += loss
    print({'epoch': epoch, 'loss': loss.item()})

pkl_filename = 'model.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

print('Done!')
