import os
import json
import numpy as np
import spacy
import random
import pickle
import torch

from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

nlp = spacy.load("en_core_web_sm")
lemmatizer = nlp.vocab.morphology.lemmatizer


dataset = []
for file_name in ['file1.json', 'file2.json', 'file3.json','file4.json','file5.json','file6.json']:
    with open(file_name, 'r') as f:
        dataset.extend(json.load(f))

# Load the files
file1 = dataset(os.path.abspath(os.path.join( "file1.json")))
file2 = dataset(os.path.abspath(os.path.join( "file2.json")))
file3 = dataset(os.path.abspath(os.path.join( "file3.json")))
file4 = dataset(os.path.abspath(os.path.join( "file4.json")))
file5 = dataset(os.path.abspath(os.path.join( "file5.json")))
file6 = dataset(os.path.abspath(os.path.join("file6.json")))

# Select the files to be used for training and concatenate them
all_Files = [file1, file3, file4, file6]
words = []
labels = []
documents = []
ignore_words = ['?', '!']

for data in all_Files:
    for intent in data:
        if len(intent['tags']) == 0:
            tag = "unspecified"
        else:     
            ##Extracting only the first tags as they're the most relevant
            tag = intent['tags'][0]
            question = intent["question"]
            wrds = [token.text for token in nlp(question)]
    
            words.extend(wrds)
            documents.append((wrds, tag))
            
            if tag not in labels:
                labels.append(tag)
                
words = [lemmatizer(w, 'NOUN')[0].lower() for w in words if w not in ignore_words]
words = sorted(list(set(words)))

labels = sorted(list(set(labels)))

print (len(documents), "documents")
print (len(labels), "labels", labels)
print (len(words), "unique lemmatized words", words)

'''
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(labels, open('labels.pkl','wb'))

class QADataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = torch.tensor(self.data[index][0])
        y = torch.tensor(self.data[index][1])
        return x, y
    
training = []
out_empty = [0 for _ in range(len(labels))]
for doc in documents:
    bag = []
    
    pattern_words = doc[0]
    pattern_words = [lemmatizer(w, 'NOUN')[0].lower() for w in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        

    output_row = out_empty[:]
    output_row[labels.index(doc[1])] = 1

    training.append([bag, output_row])
    
random.shuffle(training)
train_data = QADataset(training)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
print("Training data created")

class QAModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(QAModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc
'''