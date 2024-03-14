# Import libraries.
import spacy
import pandas as pd
from textblob import TextBlob

# Load the spaCy model.
nlp = spacy.load('en_core_web_sm')

# Load the dataset into a pandas dataframe.
df = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# Remove missing values.
clean_data = df.dropna(subset=['reviews.text'])


# Function to remove stopwords.
def remove_stopwords(text):
    """
    Removes stopwords from the given text.

    Parameters:
    text (str): The input text for stopword removal.

    Returns:
    str: The input text with stopwords removed.
    """
    # Convert text to lowercase and tokenize it using spaCy.
    doc = nlp(str(text.lower()))

    # Filter out stopwords and join the remaining tokens.
    tokens = [token.text for token in doc if not token.is_stop]
    return ' '.join(tokens)


# Function to perform sentiment analysis.
def sentiment(review_text):
    """
    Analyses the sentiment of a given review text.

    Parameters:
    review_text (str): The text of the review to be analysed.

    Returns:
    Tuple: (Positive, Negative, or Neutral).
    A tuple containing the sentiment response. 
    Sentiment Value:
    The sentiment value (a float representing the polarity).
    """
    # Analyse sentiment using TextBlob.
    blob = TextBlob(review_text)
    sentiment_value = blob.sentiment.polarity

    # Determine sentiment response based on sentiment value.
    if sentiment_value < 0:
        sentiment_response = 'Negative'
    elif sentiment_value > 0:
        sentiment_response = 'Positive'
    else:
        sentiment_response = 'Neutral'
    return sentiment_response, sentiment_value


# Input the index of the review to analyse from the dataset.
while True:
    review_index = input("\nEnter a review number (0 - 34659)."
                    "\nInput 'exit' or 'quit' to end the program: ")
    if review_index.lower() in ['exit', 'quit']:
        break
    try:
        review_index = int(review_index)
        review_text = clean_data.loc[review_index, 'reviews.text']
        processed_review = remove_stopwords(review_text)
        sentiment_response, sentiment_value=sentiment(processed_review)
        
        # Print the selected review with sentiment analysis output.
        print("\nSelected Review:")
        print(review_text)
        print("\nSentiment:", sentiment_response)
        print("Sentiment Value:", sentiment_value)
    except (ValueError, KeyError):
        print("\nInvalid input.")