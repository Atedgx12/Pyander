import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Importing Support Vector Machine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import os
import joblib
from docx import Document
import re
from transformers import BertTokenizer, AutoModel # import it!
import nlpaug.augmenter.word as naw
import albumentations as A
import ensemble 
import sys 


sys.path.append('C:\\Users\\Owner\\Desktop\\Pyander')

def load_local_data(file_path, encoding="utf-8", delimiter=",", column_names=["text", "label"]):
    """
    Load local data.

    Args:
        file_path (str): The path to the CSV file.
        encoding (str, optional): The encoding of the text data. Defaults to "utf-8".
        delimiter (str, optional): The delimiter of the CSV file. Defaults to ",".
        column_names (list, optional): The column names of the CSV file. Defaults to ["text", "label"].

    Returns:
        A pandas dataframe.
    """

    # Check if the file exists.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the data into a pandas dataframe.
    data = pd.read_csv(
        file_path,
        encoding=encoding,
        delimiter=delimiter,
        names=column_names,
    )

    # Generate the labels.
    data["label"] = generate_number_sequence(len(data))

    return data


def load_and_preprocess_data():
    """
    Load and preprocess the data.

    Returns:
        A tuple of (X_train_text, X_train_images, y_train).
    """

    # Load the data.
    data = load_local_data("data.csv")

    # Preprocess the text data.
    X_train_text = preprocess_text(data["text"])

    # Preprocess the image data.
    X_train_images = preprocess_images(data["image"])

    # Extract the labels.
    y_train = data["label"]

    return X_train_text, X_train_images, y_train


# Load and preprocess your data
X_train_text, X_train_images, y_train = load_and_preprocess_data()

# Define the pair of classes for augmentation
class_a = 0
class_b = 1

# Train the ensemble model
ensemble_model = augmentation_ensemble.train_ensemble(X_train_text, y_train, X_train_images, y_train_images, class_a, class_b)

# Evaluate the ensemble model and perform predictions
accuracy = evaluate_ensemble(ensemble_model, X_test, y_test)
predictions = ensemble_model.predict(X_test)
def perform_text_augmentation(X_train, y_train, class_a, class_b):
    augmenter = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
    
    augmented_X_train = []
    augmented_y_train = []
    
    for text, label in zip(X_train, y_train):
        if label in [class_a, class_b]:
            augmented_text = augmenter.augment(text)
            augmented_X_train.append(augmented_text)
            augmented_y_train.append(label)
    
    return augmented_X_train, augmented_y_train
def train_ensemble(X_train, y_train, X_train_images, y_train_images, class_a, class_b):
    augmented_X_train, augmented_y_train = perform_text_augmentation(X_train, y_train, class_a, class_b)
    augmented_X_train_images, augmented_y_train_images = perform_image_augmentation(X_train_images, y_train_images, class_a, class_b)
    
    model_text = LogisticRegression()
    model_text.fit(augmented_X_train, augmented_y_train)
    
    model_image = YourImageClassifier()
    model_image.fit(augmented_X_train_images, augmented_y_train_images)
    
    ensemble = VotingClassifier(estimators=[
        ('model_text', model_text),
        ('model_image', model_image),
    ], voting='soft')
    
    ensemble.fit(X_train, y_train)
    
    return ensemble
    
def perform_image_augmentation(X_train_images, y_train_images, class_a, class_b):
    aug = A.Compose([
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ], p=1)
    
    augmented_X_train_images = []
    augmented_y_train_images = []
    
    for image, label in zip(X_train_images, y_train_images):
        if label in [class_a, class_b]:
            augmented_image = augment_image(image)
            augmented_X_train_images.append(augmented_image)
            augmented_y_train_images.append(label)
    
    return augmented_X_train_images, augmented_y_train_images

def train_ensemble(X_train, y_train, X_train_images, y_train_images, class_a, class_b):
    augmented_X_train, augmented_y_train = perform_text_augmentation(X_train, y_train, class_a, class_b)
    augmented_X_train_images, augmented_y_train_images = perform_image_augmentation(X_train_images, y_train_images, class_a, class_b)
    
    model_text = LogisticRegression()
    model_text.fit(augmented_X_train, augmented_y_train)
    
    model_image = YourImageClassifier()
    model_image.fit(augmented_X_train_images, augmented_y_train_images)
    
    ensemble = VotingClassifier(estimators=[
        ('model_text', model_text),
        ('model_image', model_image),
    ], voting='soft')
    
    ensemble.fit(X_train, y_train)
    
    return ensemble


from langchain.llms import OpenAI
llm = OpenAI(openai_api_key="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")

# Initialize a Logistic Regression model for trinary decisions
decision_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Initialize the transformer model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_bert = AutoModel.from_pretrained("bert-base-uncased")

# Initialize an SVM model for trinary decisions
svm_decision_model = SVC()  # Added SVM model

# Initialize a DataFrame to store features and labels for decision making
decision_data = pd.DataFrame(columns=['feature1', 'feature2', 'decision'])

user_performance = []
difficulty_levels = []

X = ["Thisisapositivesentence", "Thisisaneutralsentence", "Thisisanegativesentence"]
y = ["positive", "neutral", "negative"]

# Initialize an empty DataFrame to store interactions
interaction_df = pd.DataFrame(columns=['user_input', 'ai_output'])

vectorizer = CountVectorizer()
model = MultinomialNB()
# -----------------------------Chunk 1--------------------------------------------------------
# Function to prepare features using BERT
def prepare_features_with_transformer():
    global interaction_df
    inputs = tokenizer(list(interaction_df['user_input']), padding=True, truncation=True, return_tensors="dict")
    outputs = model_bert(**inputs)
    X = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return X

# Function to train the decision model using SVM
def train_decision_model(features, labels):
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))  # Using Support Vector Classifier with Standard Scaler
    model.fit(features, labels)
    return model
    
    
# Function to predict the next action
def predict_next_action(model, vectorizer, label_encoder, text):
    X_vectorized = vectorizer.transform([text])
    prediction = model.predict(X_vectorized)
    label = label_encoder.inverse_transform(prediction)[0]
    return label

# Function to train the difficulty model using SVM
def train_difficulty_model():
    X = np.array(user_performance).reshape(-1, 1)
    y = difficulty_levels
    model = SVC()  # Using Support Vector Classifier
    model.fit(X, y)
    return model

# Function for a mental stimulating activity with polarized decision options
def mental_stimulus(difficulty_model=None):
    if difficulty_model:
        pred_difficulty = difficulty_model.predict(np.array([[sum(user_performance) / len(user_performance)]]))[0]
    else:
        pred_difficulty = random.choice(['easy', 'medium', 'hard'])

    # Polarized decision options based on difficulty
    if pred_difficulty == 'easy':
        num1, num2 = random.randint(1, 10), random.randint(1, 10)
    elif pred_difficulty == 'medium':
        num1, num2 = random.randint(10, 50), random.randint(10, 50)
    else:  # hard
        num1, num2 = random.randint(50, 100), random.randint(50, 100)

    print(f"What is {num1} + {num2}?")
    answer = int(input("Your answer: "))
    correct = answer == (num1 + num2)

    user_performance.append(int(correct))
    difficulty_levels.append(pred_difficulty)

    return correct

def prepare_features():
    global interaction_df
    X = vectorizer.fit_transform(interaction_df['user_input'])
    return X

def capture_interaction(user_input, ai_output):
    global interaction_df
    new_row = {'user_input': user_input, 'ai_output': ai_output}
    interaction_df = interaction_df.append(new_row, ignore_index=True)


# This code generates the data for the file `data.csv`.
def generate_data(num_samples):
  """
  Generates data for the file `data.csv`.

  Args:
    num_samples: The number of samples to generate.

  Returns:
    A list of lists, where each inner list contains a text and a label.
  """

  data = []
  for _ in range(num_samples):
    text = random.choice(["positive", "negative", "neutral"])
    label = "Positive" if text == "positive" else "Negative" if text == "negative" else "Neutral"
    data.append([text, label])

  return data
  
def initialize_google_api(api_key, cse_id):
    service = build("customsearch", "v1", developerKey=api_key)
    return service

  
def fetch_data_from_google(service, query, cse_id):
    res = service.cse().list(q=query, cx=cse_id).execute()
    return res['items']


def train_decision_model(features, labels):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(features, labels)
    return model
    
def classify_sentiment(user_input):
    """
    Classifies the sentiment of the user's input.

    Args:
        user_input: The user's input.

    Returns:
        The sentiment of the user's input.
    """

    # Try to convert the user input to a number.
    try:
        user_input = int(user_input)
    except ValueError:
        # If the user input is not a number, skip sentiment analysis and respond directly.
        return respond_to_user_input(user_input)

    # Classify the sentiment of the user input.
    if user_input > 0:
        return "positive"
    elif user_input == 0:
        return "neutral"
    else:
        return "negative"

def create_data_file():
  """
  Creates the file `data.csv` and populates it with some sample data.
  """

  # Create a pandas dataframe.
  df = pd.DataFrame({
    "text": ["This is a sentence.", "This is another sentence."]
  })

  # Write the dataframe to a CSV file.
  df.to_csv("data.csv")

def preformat_data(data):
  """
  Preformats the data to be in a proper format.

  Args:
    data: The data to be preformatted.

  Returns:
    A preformatted version of the data.
  """

  # Convert the data to a list of strings.
  data = [str(item) for item in data]

  # Remove all punctuation from the data.
  for i in range(len(data)):
    data[i] = re.sub(r"[^\w\s]", "", data[i])

  # Remove all whitespace from the data.
  for i in range(len(data)):
    data[i] = re.sub(r"\s+", "", data[i])

  # Return the preformatted data.
  return data
  
  
user_inputs = ["Hello, world!", "How are you?", "What's up?"]

def prepare_features_with_transformer():
    global interaction_df
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Initialize tokenizer
    inputs = tokenizer(list(interaction_df['user_input']), padding=True, truncation=True, return_tensors="pt")
    outputs = model_bert(**inputs)
    X = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return X


# Load the data into a pandas dataframe.
df = pd.read_csv("data.csv")

# Preformat the data.
data = preformat_data(df["text"])

# Vectorize the text data
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Label encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into a training set and a test set.
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.75)

# Train a logistic regression model on the training set.
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set.
score = model.score(X_test, y_test)

# Train a decision model using SVM on the training set.
decision_model = train_decision_model(X_train, y_train)

# Evaluate the model on the test set.
score = decision_model.score(X_test, y_test)

print(f"The accuracy score is {score:.2f}")

##--------------------------------Chunk 2-------------------------------------------------
def generate_number_sequence(length):
    sequence = []
    for i in range(length):
        sequence.append(i + 1)
    return sequence
      
def remove_special_characters(text):
    """
    Remove special characters from a string.

    Args:
        text (str): The string to remove special characters from.

    Returns:
        str: The string with special characters removed.
    """

    special_characters = ["\n", "\t", " "]
    for special_character in special_characters:
        text = text.replace(special_character, "")

    return text




def check_file_exists(file_path):
    if not file_path:
        return False

    try:
        return os.path.isfile(file_path)
    except FileNotFoundError:
        return False

def load_data(data_source):
    data = []

    if data_source == 'local':
        num_files = int(input('Enter the number of local files: '))
        file_paths = [input(f'Enter the path to local file {i+1}: ').strip('\"') for i in range(num_files)]

        for file_path in file_paths:
            file_extension = file_path.split('.')[-1]
            data.append(load_local_data(file_extension, file_path))

    elif data_source == 'web':
        url = input('Enter the URL of the webpage to scrape: ')
        # Implement web data loading logic
        pass
    else:
        raise ValueError('Invalid data source')

    if len(data) == 0:
        raise ValueError('No data found')

    return pd.concat(data, ignore_index=True)

def preprocess_data(data):
    vectorizer = CountVectorizer(max_features=10000)
    vectorizer.fit(data['text'])
    X = vectorizer.transform(data['text'])
    vocabulary = vectorizer.get_feature_names_out()
    return X, vocabulary

def split_data(X, y):
  """
  Splits the data into a training set and a test set, using stratified splitting.

  Args:
    X: The data to split.
    y: The labels for the data.

  Returns:
    The training set, the test set, the training labels, and the test labels.
  """

  # Create a label encoder.
  encoder = LabelEncoder()

  # Fit the label encoder to the labels.
  encoder.fit(y)

  # Transform the labels.
  y = encoder.transform(y)
  
   # Convert the data to numeric data.
  X = encoder.transform(X)
  

  # Split the data into a training set and a test set, using stratified splitting.
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
  
 

  return X_train, X_test, y_train, y_test

# Your existing train_model function
def train_model(data, model_type="MultinomialNB", continuous_learning=True):
    if continuous_learning:
        global interaction_df
        interaction_df = pd.concat([interaction_df, data])

    if model_type == "MultinomialNB":
        model = MultinomialNB()
    elif model_type == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif model_type == "SVC":
        model = SVC(gamma='auto')
    else:
        raise ValueError("Invalid model type")

    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(data["text"])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(data["label"])

    # Split the data into a training set and a test set.
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, stratify=y_encoded)

    model.fit(X_train, y_train)

    # Evaluate the model on the test set.
    score = model.score(X_test, y_test)

    return model, score

# Initialize an empty DataFrame for interaction logging
interaction_df = pd.DataFrame(columns=['text', 'label'])

while True:  # Keep running the program
    # Generate simulated user input and label using the model
    simulated_user_input = "Your prediction code here"  # Replace with your prediction code
    simulated_label = "Your prediction code here"  # Replace with your prediction code

    print(f"Simulated You: {simulated_user_input}")
    print(f"Simulated Label: {simulated_label}")

    # Log the interaction
    new_data = pd.DataFrame({'text': [simulated_user_input], 'label': [simulated_label]})
    interaction_df = pd.concat([interaction_df, new_data])

    # Train or update the model
    model = train_model(interaction_df, continuous_learning=True)

    # ... (rest of your code remains the same)

    # Generate a decision to continue or not using the model
    continue_decision = "Your prediction code here"  # Replace with your prediction code

    if continue_decision.lower() != 'yes':
        print("Exiting the program.")
        break

if __name__ == "__main__":
    main()



def save_model(model, file_path):
    joblib.dump(model, file_path)

def load_model(file_path):
    model = joblib.load(file_path)
    return model
difficulty_model = None









def main():
    # Initialize Google API
    api_key = "AIzaSyDD6UFLX91lWDtvik098FbS6-f0huNENkg"
    cse_id = "72a9f7da7e1774bcf"
    service = initialize_google_api(api_key, cse_id)
    
    # Initialize an empty DataFrame for decision logging    
    decision_df = pd.DataFrame(columns=['accuracy', 'num_interactions', 'decision'])

    # Initialize decision model
    decision_model = None
    
    # Step 1: Data Collection
    contexts = ["low_accuracy", "high_accuracy", "new_data"]
    questions = ["Would you like to provide more training data?", 
             "Would you like to test the model?", 
             "Would you like to label the new data?"]

    # Step 2: Feature Engineering
    vectorizer = CountVectorizer()
    label_encoder = LabelEncoder()
    model = None

    # Step 3: Model Training
    clf = MultinomialNB()
    clf.fit(X, questions)

# Step 4: Integration
def get_question(context):
    context_vectorized = vectorizer.transform([context])
    question = clf.predict(context_vectorized)
    return question[0]


    try:
        # Step 1: Load Data
        data_source = input("Enter the data source (local/web): ")
        data = load_data(data_source)
        
        # Step 2: Preprocess Data
        data["text"] = preformat_data(data["text"])
        
        # Step 3: Feature Extraction
        vectorizer = CountVectorizer()
        X_vectorized = vectorizer.fit_transform(data["text"])
        
        # Step 4: Label Encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(data["label"])
        
        # Check the number of unique labels
        unique_labels = len(set(y_encoded))
        if unique_labels <= 1:
            print(f"The number of classes has to be greater than one; got {unique_labels} class(es)")
            new_labels = input("Please enter new labels for the data, separated by commas: ").split(',')
            
            # Validate the number of new labels
            if len(new_labels) != len(data):
                print("The number of new labels doesn't match the length of the index.")
                raise ValueError("Invalid number of new labels.")
            
            # Update labels and re-encode
            data["label"] = new_labels
            y_encoded = label_encoder.fit_transform(data["label"])
        
        # Step 5: Split Data
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2)
        
        # Step 6: Train Model
        model_type = input("Enter the model type (MultinomialNB/RandomForestClassifier/SVC): ")
        model = train_model(data, model_type)
        
        # Step 7: Evaluate Model
        score = model.score(X_test, y_test)
        print(f"The accuracy score is {score:.2f}")

    except ValueError as e:
        print(f"An error occurred: {e}")

        # Fetch more data if needed
        user_input = input("Would you like to fetch more data to train the model? (yes/no): ")
        if user_input.lower() == 'yes':
            query = input("Enter the query to search for more data: ")
            fetched_data = fetch_data_from_google(service, query)
            # Process fetched_data to add more data to your training set

    # Ask the user (or the system) if they want to continue
    simulated_continue = generate_continue_decision(model_for_continue_decision)  # Implement this function
    print(f"Simulated Continue: {simulated_continue}")
    if simulated_continue.lower() != 'yes':
        print("Exiting the program.")

if __name__ == "__main__":
    main()
