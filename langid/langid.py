import fasttext
import re
from collections import Counter
languages = {
    "0": "sma",
    "1": "sme", 
    "2": "smj",
    "3": "fin",
    "4": "est"
}
class WordLid:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)
        self.threshold = 0.5

    def set_threshold(self, threshold):
        self.threshold = threshold

    def _clean_word(self, word):
        word = word.lower()
        word = re.sub(r'[^\w\s]', '', word)
        word = re.sub(r'\s+', ' ', word)
        word = re.sub(r'\d', '', word)
        return word.strip()

    def _predict_all_languages(self, word):
        cleaned_word = self._clean_word(word)
        labels, probabilities = self.model.predict(cleaned_word, k=-1)
        
        print(word)
        for l, p in zip(labels, probabilities):
            print(f'{languages[l.replace("__label__", "")]} {p:.4f}')
        return {label.replace('__label__', ''): prob for label, prob in zip(labels, probabilities)}

    def _get_main_language(self, text):
        words = [self._clean_word(word) for word in text.split() if word]
        language_counts = Counter(
            max(self._predict_all_languages(word), key=self._predict_all_languages(word).get)
            for word in words if self._predict_all_languages(word)
        )
        return language_counts.most_common(1)[0][0] if language_counts else None


    def get_lang_array(self, text):
        main_language = self._get_main_language(text)
        if main_language is None:
            return ['unk'] * len(text)

        lang_array = [main_language] * len(text)
        word_start_index = 0

        for word in text.split():
            word_start_index = text.find(word, word_start_index)
            cleaned_word = self._clean_word(word)
            if not cleaned_word:
                word_start_index += len(word)
                continue

            predictions = self._predict_all_languages(cleaned_word)
            if not predictions:
                word_start_index += len(word)
                continue

            best_word_lang = max(predictions, key=predictions.get)
            main_lang_prob_for_word = predictions.get(main_language, 0.0)  # Get main lang prob *for this word*
            best_lang_prob = predictions[best_word_lang]

            # Key change: Check if the best language probability is 0.5 greater than the main language probability *for this word*
            if best_lang_prob >= main_lang_prob_for_word + 0.5:
                for i in range(len(word)):
                    lang_array[word_start_index + i] = best_word_lang

            word_start_index += len(word)
        return [int(x) for x in lang_array]
        #return lang_array

if __name__ == '__main__':
    model_path = 'lang_id_model_q.bin'
    identifier = WordLid(model_path)

    test_texts = [
        "Mumenvákki ođđasamos badji ii gávdno vuos sámegillii. Áigumuššan lea goit dubbet dan maiddái sámegielaide, lohká Yle Draama hoavda Jarmo Lampela."
     
    ]

    for text in test_texts:
        lang_array = identifier.get_lang_array(text)
        print(f"\nText: '{text}'")
        print(f"Language Array: {lang_array}")

        #for i in range(0,len(text)):
        #    print(text[i], lang_array[i])
        assert len(lang_array) == len(text), "Length mismatch!"


    # Example of changing the threshold:
    identifier.set_threshold(0.8)
    lang_array = identifier.get_lang_array("Bonjour le monde!")
    print(f"\nText: 'Bonjour le monde!' (with threshold 0.8)")
    print(f"Language Array: {lang_array}")
