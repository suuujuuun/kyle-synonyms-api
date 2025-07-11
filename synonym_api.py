from gensim.models import KeyedVectors
import os, json
from flask import Flask, request, jsonify

app = Flask(__name__)

# ───────────────────────── ① 경로 / 환경변수 ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLOVE_PATH  = os.getenv("GLOVE_PATH",  os.path.join(BASE_DIR, "glove.6B.100d.txt"))
WORDS_PATH  = os.getenv("WORDS_PATH",  os.path.join(BASE_DIR, "cleaned_words_final_filtered.txt"))

# ───────────────────────── ② 리소스 로딩 (import 시 1회) ────────────
print("⇢ Loading GloVe…")
word_vectors = KeyedVectors.load_word2vec_format(GLOVE_PATH, binary=False,
                                                 no_header=True, encoding="latin-1")
print("✓ GloVe loaded.")

print("⇢ Loading custom-word list…")
with open(WORDS_PATH) as f:
    custom_words = {w.strip().lower() for w in f}
print(f"✓ {len(custom_words):,} custom words loaded.")

# ───────────────────────── ③ 라우트 ──────────────────────────────
@app.route("/")
def healthcheck():
    return jsonify({"status": "ok"}), 200          # Render health check 용

@app.route("/synonyms")
def synonyms():
    word = request.args.get("word", "").lower()
    if not word:
        return jsonify({"error": "parameter 'word' required"}), 400
    if word not in word_vectors:
        return jsonify({"error": f"'{word}' not in GloVe vocab"}), 404

    sims = word_vectors.most_similar(word, topn=10_000)
    results = [
        {"word": w, "score": s}
        for w, s in sims
        if w.lower() in custom_words
    ][:50]
    return jsonify(results)
