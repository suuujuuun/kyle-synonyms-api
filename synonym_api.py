from flask import Flask, request, jsonify
import random
import json
from gensim.models import KeyedVectors
import os

app = Flask(__name__)

# Global variables for GloVe model and custom words
word_vectors = None
custom_words = None

# Paths to your files
GLOVE_FILE_PATH = 'glove.6B.100d.txt'
CUSTOM_WORDS_FILE_PATH = 'cleaned_words_final_filtered.txt'
def load_resources():
    global word_vectors, custom_words
    if word_vectors is None:
        print("Loading GloVe model...")
        word_vectors = KeyedVectors.load_word2vec_format(GLOVE_FILE_PATH, binary=False, no_header=True, encoding='latin-1')
        print("GloVe model loaded.")
    
    if custom_words is None:
        print("Loading custom words...")
        with open(CUSTOM_WORDS_FILE_PATH, 'r') as f:
            custom_words = set(line.strip().lower() for line in f)
        print("Custom words loaded.")

@app.route('/synonyms', methods=['GET'])
def get_synonyms():
    word = request.args.get('word')
    if not word:
        return jsonify({"error": "Word parameter is missing"}), 400

    load_resources() # Ensure resources are loaded

    try:
        if word.lower() not in word_vectors:
            return jsonify({"error": f"Input word '{word}' not found in GloVe vocabulary."}), 404

        similar_words_candidates = word_vectors.most_similar(word.lower(), topn=10000) # Increased topn
        
        filtered_candidates = []
        for similar_word, score in similar_words_candidates:
            if similar_word.lower() in custom_words:
                filtered_candidates.append({"word": similar_word, "score": score})

        print(f"Found {len(filtered_candidates)} filtered candidates for '{word}'") # DEBUG PRINT

        num_words_to_return = 50 

        num_words_to_return = 50 
        final_words = filtered_candidates[:num_words_to_return]

        return jsonify(final_words)

    except KeyError as ke:
        return jsonify({"error": f"KeyError: {ke}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    # Load resources when the app starts
    load_resources()
    app.run(host='0.0.0.0', port=5001, debug=True)