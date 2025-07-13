
import json
from collections import Counter

# This is a simplified heuristic for word complexity.
# A more sophisticated approach might use NLP libraries like NLTK or spaCy
# to get word frequencies from a large corpus, or use pre-defined word lists.
def get_word_level(word):
    """Categorizes a word into beginner, intermediate, or advanced."""
    length = len(word)
    if length <= 4:
        return "beginner"
    elif length <= 7:
        return "intermediate"
    else:
        return "advanced"

def label_words(input_file, core_output_file, levels_output_file):
    """
    Reads words, identifies core words, categorizes them by difficulty,
    and saves them to JSON files.
    """
    with open(input_file, 'r') as f:
        words = [line.strip() for line in f if line.strip()]

    # Determine core words based on frequency (top 5000 most common)
    # In a real-world scenario, we'd use a proper frequency list.
    # For this example, we'll consider the first 5000 words in the list as "core"
    # as they appear to be sorted by some frequency metric already.
    core_words = words[:5000]

    word_levels = {
        "beginner": [],
        "intermediate": [],
        "advanced": []
    }

    for word in words:
        level = get_word_level(word)
        word_levels[level].append(word)

    # Save core words to JSON
    with open(core_output_file, 'w') as f:
        json.dump(core_words, f, indent=4)
    print(f"Saved {len(core_words)} core words to {core_output_file}")

    # Save leveled words to JSON
    with open(levels_output_file, 'w') as f:
        json.dump(word_levels, f, indent=4)
    print(f"Saved leveled words to {levels_output_file}")
    print(f" - Beginner: {len(word_levels['beginner'])} words")
    print(f" - Intermediate: {len(word_levels['intermediate'])} words")
    print(f" - Advanced: {len(word_levels['advanced'])} words")


if __name__ == '__main__':
    INPUT_FILE = '/Users/seungjun/Desktop/API/Kylesy/cleaned_words_final_filtered.txt'
    CORE_OUTPUT_FILE = '/Users/seungjun/Desktop/API/Kylesy/core_words.json'
    LEVELS_OUTPUT_FILE = '/Users/seungjun/Desktop/API/Kylesy/word_levels.json'
    label_words(INPUT_FILE, CORE_OUTPUT_FILE, LEVELS_OUTPUT_FILE)
