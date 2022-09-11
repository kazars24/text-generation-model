import argparse
import os
import nltk
import nltk.tokenize
import torch
import pickle

script = os.path(__file__)

parser = argparse.ArgumentParser(prog=script.name)
parser.add_argument('-m', '--model', required=True, help='Path to model')
parser.add_argument('-p', '--prefix', required=True, help='Some words')
parser.add_argument('-l', '--length', required=True, help='Length of generated sentence')
args = parser.parse_args()

model_path = args.model
prefix = args.prefix
length = args.length

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Generate on GPU')
else:
    print('Generate on CPU')

text_sample = ''
for file in os.listdir('data'):
    with open('data' + '/' + file, encoding='utf-8') as text_file:
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

with open(model_path, 'rb') as file:
    model = pickle.load(file)


def evaluate(prime_str='вовочка говорит', predict_len=100, temperature=0.8):
    if train_on_gpu:
        hidden = model.init_hidden().cuda()
    else:
        hidden = model.init_hidden()

    for p in range(predict_len):
        if train_on_gpu:
            prime_input = torch.tensor([word_to_ix[w] for w in prime_str.split()], dtype=torch.long).cuda()
        else:
            prime_input = torch.tensor([word_to_ix[w] for w in prime_str.split()], dtype=torch.long)
        inp = prime_input[-2:]  # last two words as input
        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted word to string and use as next input
        predicted_word = list(word_to_ix.keys())[list(word_to_ix.values()).index(top_i)]
        prime_str += " " + predicted_word
    #         inp = torch.tensor(word_to_ix[predicted_word], dtype=torch.long)

    return prime_str


print(evaluate(prefix, length, temperature=0.8))
