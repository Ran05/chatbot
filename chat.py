#The code you provided is a Python script that demonstrates a simple chatbot using a pre-trained neural network model. Here's a breakdown of how the code works:

# 1. It imports the necessary libraries and modules, including random, json, torch, the NeuralNet model class from model.py, and helper 
# functions from nltk_utils.py.

# 2. It checks the availability of a GPU for training the model and sets the device accordingly.
# 3. It loads the intents data from a JSON file (intents.json) and the trained model data from a binary file (data.pth).
# 4. It retrieves the necessary information from the loaded model data, including input size, hidden size, output size, all words, 
# tags, and the model state.

# 5.It initializes an instance of the NeuralNet model, loads the state dictionary into it, and sets the model in evaluation mode.
# 6.It defines the name of the chatbot as "Sam" and defines a function get_response(msg) to generate a response given a user message.
# 7.Inside the get_response function, it tokenizes the user message, converts it into a bag-of-words representation using the bag_of_words function, 
# and prepares the input tensor for the model.

# 8.It feeds the input tensor to the model and obtains the output, performs a softmax operation on the output to get the predicted probabilities,
#  and determines the predicted tag and its corresponding probability.

# 9.If the probability is greater than 0.75, it iterates through the intents and selects a random response from the intent matching the predicted tag.
# 10. If the probability is not greater than 0.75 or no matching intent is found, it returns the default response "I do not understand...".
# 11. Finally, it runs an infinite loop that prompts the user for input, calls the get_response function to get the chatbot's response, 
# and prints the response. The loop continues until the user enters "quit".

# 10. To use this code, make sure you have the required files (intents.json, data.pth, model.py, nltk_utils.py) in the same directory, and then run the script.
#  You can interact with the chatbot by entering messages in the terminal.



import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    # If the probability is greater than 0.75, it iterates through the intents 
    # and selects a random response from the intent matching the predicted tag

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Please specify your questions."
    #  If the probability is not greater than 0.75 or no matching intent is found,
    #  it returns the default response "I do not understand...".

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

