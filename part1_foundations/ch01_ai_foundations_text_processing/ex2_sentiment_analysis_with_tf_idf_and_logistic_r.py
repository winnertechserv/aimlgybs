"""
Chapter 1, Example 2: Sentiment Analysis with TF-IDF and Logistic Regression
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
TF-IDF turns each review into a vector of word weights, and logistic regression learns
which words point to 'positive' and which to 'negative'. We also include word pairs
(ngram_range=(1, 2)). That lets the model learn 'not good' and 'not bad' as separate
clues, so it correctly separates the two tricky sentences from the overview. The last
lines show the strongest signals the model learned for each class.

Run it:           python ex2_sentiment_analysis_with_tf_idf_and_logistic_r.py
Expected output:  expected_output/ex2_sentiment_analysis_with_tf_idf_and_logistic_r.txt
"""
# Chapter 1 - Example 2: Text classification (sentiment analysis)
# Words -> numbers with TF-IDF, then a classifier learns which words signal which mood.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_texts = [
    "the food was good", "really good service and tasty food", "great place loved it",
    "the staff were friendly and helpful", "excellent meal will come again",
    "not bad at all, quite good", "the pizza was delicious", "lovely ambience and good music",
    "the food was bad", "terrible service and cold food", "awful place hated it",
    "the staff were rude and slow", "worst meal ever never again",
    "not good at all, quite bad", "the pizza was burnt", "noisy and dirty, very bad",
]
train_labels = ["pos"] * 8 + ["neg"] * 8

# ngram_range=(1,2) also keeps word PAIRS like "not good" and "not bad"
vec = TfidfVectorizer(ngram_range=(1, 2))
X = vec.fit_transform(train_texts)
clf = LogisticRegression(C=10, max_iter=1000).fit(X, train_labels)
print("Vocabulary size (words + word pairs):", len(vec.vocabulary_))

tests = ["The food was good, not bad at all",
         "The food was bad, not good at all",
         "friendly staff and delicious pizza",
         "rude staff and cold pizza"]
for t in tests:
    p = clf.predict_proba(vec.transform([t]))[0]
    label = clf.classes_[p.argmax()]
    print(f"{t:38s} -> {label}  (P(pos) = {p[list(clf.classes_).index('pos')]:.2f})")

# Which features push hardest towards each class?
names = vec.get_feature_names_out()
w = clf.coef_[0]            # positive weight -> class 'pos' (classes_ = ['neg','pos'])
order = w.argsort()
print("\nStrongest NEGATIVE signals:", [names[i] for i in order[:5]])
print("Strongest POSITIVE signals:", [names[i] for i in order[-5:][::-1]])
