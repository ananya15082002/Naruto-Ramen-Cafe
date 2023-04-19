'''import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset = []
for file_name in ['file1.json', 'file2.json', 'file3.json','file4.json','file5.json','file6.json']:
    with open(file_name, 'r') as f:
        dataset.extend(json.load(f))

# Extract the questions and answers from the dataset
questions = [data["question"] for data in dataset]
answers = [data["answer"] for data in dataset]

# Encode the questions and answers using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)
y = answers

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier on the training set
classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)




# Define a function to search for the most relevant question based on the input tags
def search_questions(tags):
    relevant_indices = []
    for i, data in enumerate(dataset):
        if set(tags).issubset(set(data["tags"])):
            relevant_indices.append(i)
    return relevant_indices

# Define a function to predict the answer for a given question and tags
def predict_answer(question, tags):
    relevant_indices = search_questions(tags)
    if not relevant_indices:
        return "Sorry, no relevant questions found."
    X = vectorizer.transform([dataset[i]["question"] for i in relevant_indices])
    y_pred = classifier.predict(X)
    best_answer_index = y_pred.argmax()
    return dataset[relevant_indices[best_answer_index]]["answer"], dataset[relevant_indices[best_answer_index]]["url"]

# Take input from the user
question = input("Enter your question: ")
tags = input("Enter relevant tags (comma-separated): ").split(",")

# Predict the answer for the given question and tags
answer, url = predict_answer(question, tags)

# Print the answer and URL
print("Answer:", answer)
print("URL:", url)
'''
'''
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
dataset = []
for file_name in ['file1.json', 'file2.json']:
    with open(file_name, 'r') as f:
        dataset.extend(json.load(f))

# Extract the questions and answers from the dataset
questions = [data["question"] for data in dataset]
answers = [data["answer"] for data in dataset]
tags = [data["tags"] for data in dataset]

# Fit the vectorizer on the questions and tags separately
vectorizer_questions = TfidfVectorizer()
vectorizer_questions.fit(questions)

vectorizer_tags = TfidfVectorizer()
vectorizer_tags.fit([' '.join(tag) for tag in tags])

# Encode the questions and tags using TF-IDF vectorization
X_questions = vectorizer_questions.transform(questions)
X_tags = vectorizer_tags.transform([' '.join(tag) for tag in tags])

# Define a function to search for the most relevant question based on the input question
def search_questions(user_question):
    # Transform the user input into a TF-IDF vector
    X_user_question = vectorizer_tags.transform([user_question])
    
    # Compute the cosine similarity between the user input and all questions in the dataset
    similarity = cosine_similarity(X_user_question, X_questions).flatten()
    
    # Get the indices of the most similar questions
    most_similar_indices = np.argsort(similarity)[::-1][:5]
    
    # Get the most relevant questions and their tags
    relevant_questions = [questions[i] for i in most_similar_indices]
    relevant_tags = [tags[i] for i in most_similar_indices]
    
    # Flatten the tags list
    flat_tags = [item for sublist in relevant_tags for item in sublist]
    
    return most_similar_indices, X_user_question, relevant_questions, flat_tags

# Define a function to predict the answer for a given question and tags
def predict_answer(user_question, relevant_tags, most_similar_indices, X_user_question):
    # Encode the relevant tags using TF-IDF vectorization
    X_tags_relevant = vectorizer_tags.transform([' '.join(relevant_tags)])
    
    # Compute the cosine similarity between the relevant tags and all tags in the dataset
    tag_similarity = cosine_similarity(X_tags_relevant, X_tags).flatten()
    
    # Get the indices of the most similar tags
    most_similar_tag_indices = np.argsort(tag_similarity)[::-1][:5]
    
    # Get the most relevant questions based on both question similarity and tag similarity
    relevant_questions = []
    for i in most_similar_tag_indices:
        if i in most_similar_indices:
            relevant_questions.append(dataset[i])
    
    if not relevant_questions:
        return "Sorry, no relevant questions found."
    
    # Encode the relevant questions using TF-IDF vectorization
    X_relevant_questions = vectorizer_tags.transform([data["question"] for data in relevant_questions])
    
    # Compute the cosine similarity between the user input and the relevant questions
    question_similarity = cosine_similarity(X_user_question, X_relevant_questions).flatten()
    
    # Get the index of the most similar question
    best_question_index = np.argmax(question_similarity)
    
    # Get the answer and URL for the most similar question
    answer = relevant_questions[best_question_index]["answer"]
    url = relevant_questions[best_question_index]["url"]
    
    return answer, url


# Take user input
user_question = input("Ask a question: ")

# Search for the most relevant question based on the input question
most_similar_indices, X_user_question, relevant_questions, flat_tags = search_questions(user_question)

# Predict the answer for the most relevant question and tags
answer, url = predict_answer(user_question, flat_tags, most_similar_indices, X_user_question)

# Print the answer and URL
print("Answer: ", answer)
print("URL: ", url)

'''
import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the dataset from a JSON file
with open('file1.json', 'r') as f:
    dataset = json.load(f)

# Split the dataset into train, validation, and test sets
train_dataset = dataset[:800]
val_dataset = dataset[800:900]
test_dataset = dataset[900:]

# Load a pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Tokenize the input data and convert it to PyTorch tensors
def tokenize_data(data):
    inputs = tokenizer(data['question'], data['answer'], return_tensors='pt')
    outputs = {
        'start_positions': torch.tensor([data['answer_start']]),
        'end_positions': torch.tensor([data['answer_end']]),
        'tags': torch.tensor(data['tags']),
        'url': torch.tensor(data['url'])
    }
    return inputs, outputs

train_data = [tokenize_data(data) for data in train_dataset]
val_data = [tokenize_data(data) for data in val_dataset]
test_data = [tokenize_data(data) for data in test_dataset]

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
def train_model(train_data, val_data, model, optimizer, loss_fn, num_epochs=5):
    for epoch in range(num_epochs):
        # Train the model on the training data
        model.train()
        total_loss = 0
        for inputs, outputs in train_data:
            optimizer.zero_grad()
            loss = model(**inputs, **outputs).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_data)
        
        # Evaluate the model on the validation data
        model.eval()
        total_loss = 0
        for inputs, outputs in val_data:
            with torch.no_grad():
                loss = model(**inputs, **outputs).loss
            total_loss += loss.item()
        avg_val_loss = total_loss / len(val_data)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

train_model(train_data, val_data, model, optimizer, loss_fn)

# Test the model on new data
def predict_answer(question, context):
    inputs = tokenizer(question, context, return_tensors='pt')
    with torch.no_grad():
        start_logits, end_logits = model(**inputs).start_logits, model(**inputs).end_logits
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx]))
    return answer

def predict_tags(question):
    inputs = tokenizer(question, return_tensors='pt')
    with torch.no_grad():
        logits = model.classifier(inputs['input_ids']).squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    tags = [i for i, prob in enumerate(probs) if prob > 0.5]
    return tags

def predict_url(question):
    inputs = tokenizer
