from flask import Flask, request, jsonify
from gensim.models import KeyedVectors

app = Flask(__name__)

GLOVE_FILE_PATH = "glove.6B.100d.txt"
CUSTOM_WORDS_FILE_PATH = "cleaned_words_final_filtered.txt"

word_vectors = None
custom_words = None


def load_resources():
    global word_vectors, custom_words

    if word_vectors is None:
        print("Loading GloVe model...")
        word_vectors = KeyedVectors.load_word2vec_format(
            GLOVE_FILE_PATH, binary=False, no_header=True, encoding="latin-1"
        )
        print("GloVe model loaded.")

    if custom_words is None:
        print("Loading custom words...")
        with open(CUSTOM_WORDS_FILE_PATH) as f:
            custom_words = {line.strip().lower() for line in f}
        print("Custom words loaded.")


@app.route("/synonyms", methods=["GET"])
def get_synonyms():
    word = request.args.get("word", "").lower()
    if not word:
        return jsonify({"error": "Query parameter 'word' is required."}), 400

    try:
        if word not in word_vectors:
            return jsonify({"error": f"'{word}' not in GloVe vocabulary."}), 404

        # topn 높게 잡고 custom 단어로 필터링
        candidates = word_vectors.most_similar(word, topn=10_000)
        filtered = [
            {"word": w, "score": s}
            for w, s in candidates
            if w.lower() in custom_words
        ][:50]

        return jsonify(filtered)

    except KeyError as e:
        return jsonify({"error": f"KeyError: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


if __name__ == "__main__":
    load_resources()
    app.run(host="0.0.0.0", port=5001, debug=True)
