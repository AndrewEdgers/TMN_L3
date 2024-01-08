import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from typing import List, Tuple

# Завантаження стоп-слів і лематизатора
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Ініціалізація стоп-слів і лематизатора
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Видалення неалфавітних символів та цифр
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])

    # Токенізація
    tokens = word_tokenize(text)

    # Видалення стоп-слів і приведення до нижнього регістру
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if
              word.lower() not in stop_words and word.isalpha()]

    return tokens


# Шлях до файлу
path_to_dataset = '11-0.txt'

# Читання файлу
with open(path_to_dataset, 'r', encoding='utf-8') as file:
    text = file.read()


# Розділення тексту на глави
def divide_into_chapters(text: str, chapter_indices: List[Tuple[int, int]]):
    chapters = []
    for ((start_a, start_b), (end_a, end_b)) in zip(chapter_indices, chapter_indices[1:] + [(None, None)]):
        chapter = text[start_b:end_a] if end_a is not None else text[start_b:]
        chapters.append(chapter)
    return chapters


# Індекси початку та кінця кожного співпадіння "CHAPTER [IVXLCDM]+.\n"
chapter_indices = [(match.start(), match.end()) for match in re.finditer('CHAPTER [IVXLCDM]+.\n', text)]
chapters = divide_into_chapters(text, chapter_indices)

# Convert the set of stop words to a list
stop_words_list = list(stop_words)

# Відібрати топ 20 слів за допомогою TF-IDF
tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess_text, token_pattern=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(chapters)
feature_names = tfidf_vectorizer.get_feature_names_out()

top_words_per_chapter = {}
for chap_idx, chapter in enumerate(chapters):
    feature_index = tfidf_matrix[chap_idx, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[chap_idx, x] for x in feature_index])
    top_words = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:20]
    top_words_per_chapter[chap_idx] = [(feature_names[i], score) for i, score in top_words]

# Вивести результати для кожної глави
for chapter, words in top_words_per_chapter.items():
    print(f"Top words in chapter {chapter + 1}:")
    for word, score in words:
        print(f"{word}: {score:.4f}")
    print("\n")


# Для LDA ми потребуємо текст, розділений на слова, але не лематизований і без видалення стоп-слів
def tokenize_text(text):
    # Видалення неалфавітних символів та цифр
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])

    # Токенізація
    tokens = word_tokenize(text)

    return [word.lower() for word in tokens if word.isalpha()]


# Підготовка даних для LDA
tokenized_chapters = [tokenize_text(chapter) for chapter in chapters]

# Створення словника та корпусу для LDA
dictionary = corpora.Dictionary(tokenized_chapters)
corpus = [dictionary.doc2bow(text) for text in tokenized_chapters]

# Навчання LDA моделі
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

# Виведення тем LDA
print("\nLDA Topics with Weights:")
for idx, topic in lda_model.show_topics(formatted=False, num_topics=5, num_words=10):
    print(f"Top words in topic {idx + 1}:")
    for word, weight in topic:
        print(f"{word}: {weight:.4f}")
    print("\n")
