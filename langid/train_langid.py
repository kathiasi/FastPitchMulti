
import fasttext
import re
import random
languages = {"0":"sma",
             "1": "sme", 
             "2": "smj",
             "3": "fin",
             "4": "est"
}



def clean_word(word):
    """Cleans a word by lowercasing, removing punctuation, and extra spaces."""
    word = word.lower()
    word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
    word = re.sub(r'\s+', ' ', word)  # Remove multiple spaces
    word = re.sub(r'\d', '', word) # Remove numbers
    word = word.strip()
    word = word.lower()
    return word

def train_model(train_file, valid_file, model_path, epoch=50, lr=0.25, wordNgrams=1, dim=100):
    """Trains a FastText supervised model."""

    model = fasttext.train_supervised(
        input=train_file,
        epoch=epoch,
        lr=lr,
        minn=1,
        maxn=8,
        wordNgrams=wordNgrams,  # Capture n-grams of words (important for language ID)
        dim=dim, # Size of word vectors
        loss='softmax' # Or 'ova' (one-vs-all) if you have many languages
    )

    print(f"Model training complete.  Validation results: {model.test(valid_file)}")
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    return model


def test_model(model_path, test_file):
  """Loads and evaluates on test data."""
  model = fasttext.load_model(model_path)
  results = model.test(test_file)
  print(f"Test results: {results}")
  return results

def predict_language(model_path, word):
    """Predicts the language of a single word."""
    model = fasttext.load_model(model_path)
    cleaned_word = clean_word(word)
    label, probability = model.predict(cleaned_word)
    # The label is returned as a list of strings like ['__label__en']
    return label[0].replace('__label__', ''), probability[0]

def predict_all_languages(model, word, k=-1):
    """
    Predicts the probabilities for all languages.

    Args:
        model_path: Path to the trained FastText model.
        word: The word to predict.
        k: The number of top predictions to return.  -1 means return all.

    Returns:
        A dictionary where keys are language codes and values are probabilities.
    """
    #model = fasttext.load_model(model_path)
    cleaned_word = clean_word(word)

    # Predict with k=-1 to get all labels and probabilities
    labels, probabilities = model.predict(cleaned_word, k=k)

    # Create a dictionary for easier access
    predictions = {}
    for label, probability in zip(labels, probabilities):
        
        lang_code = label.replace('__label__', '')
        lang_code = languages[lang_code]
        predictions[lang_code] = f'{probability:.4f}'

    return predictions

if __name__ == '__main__':
    # --- Example Usage ---
    # 1. Prepare your raw data in 'raw_data.txt' (format: "lang_code word")
    #    Example:
    #    en apple
    #    fr pomme
    #    ...
    
    model_path = 'lang_id_model.bin'
    try:
        trained_model = fasttext.load_model(model_path)
    except:
        # 3. Train the model
        trained_model = train_model('lid_train.txt', 'lid_val.txt', 'lang_id_model.bin')
    
        # 4. Evaluate the model
        test_model('lang_id_model.bin', 'lid_test.txt')
        sys.exit(0)
    
    test_data = open('lid_test.txt').readlines()
    #random.shuffle(test_data)
    for l in test_data:
        lab, word = l.split()
        print(word, languages[lab[-1]], predict_all_languages(trained_model, word))
        
    
    # 5. Predict the language of a word
    word_to_predict = "Bonjour"
    predicted_language, confidence = predict_language('lang_id_model.bin', word_to_predict)
    print(f"The predicted language of '{word_to_predict}' is: {predicted_language} (confidence: {confidence:.4f})")

    word_to_predict = "apple"
    predicted_language, confidence = predict_language('lang_id_model.bin', word_to_predict)
    print(f"The predicted language of '{word_to_predict}' is: {predicted_language} (confidence: {confidence:.4f})")
