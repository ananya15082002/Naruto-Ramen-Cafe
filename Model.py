'''import json

# Load the dataset
with open('file1.json', 'r') as f:
    dataset = json.load(f)

# Define function to get answer for a given question
def get_answer(question):
    for entry in dataset:
        if entry['question'] == question:
            return entry['answer']
    return 'Sorry, I do not have an answer for that question.'

# Example usage
question = "how do i stop smoking now"
answer = get_answer(question)
print(answer)
'''

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json
# Load the BERT model and tokenizer
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load the dataset

dataset = []
for file_name in ['file1.json', 'file2.json', 'file3.json','file4.json','file5.json','file6.json']:
    with open(file_name, 'r') as f:
        dataset.extend(json.load(f))

# Define function to get answer for a given question
def get_answer(question):
    for entry in dataset:
        if entry['question'] == question:
            return entry['answer']

    # If the question is not found in the dataset, use the BERT model to find the answer
    model.eval()
    encoded_input = tokenizer(question, padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids = encoded_input['input_ids']
    token_type_ids = encoded_input['token_type_ids']
    attention_mask = encoded_input['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
    start_idx = torch.argmax(start_scores, dim=1).item()
    end_idx = torch.argmax(end_scores, dim=1).item()
    predicted_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_idx:end_idx+1]))
    return predicted_answer

question = input("Enter a question: ")
answer = get_answer(question)
print("Answer:", answer)

