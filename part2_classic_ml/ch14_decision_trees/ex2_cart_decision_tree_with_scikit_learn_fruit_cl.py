"""
Chapter 14, Example 2: CART Decision Tree with scikit-learn (Fruit Classifier)
Book: AI-Machine Learning: Get Your Base Strong (Abhinav Srivastava)

What this example shows
-----------------------
We train a real CART decision tree on a small fruit classification example. Each fruit
is described by 4 categorical features encoded as integers: Color (0=Green, 1=Yellow,
2=Red), Size (0=Small, 1=Medium, 2=Big), Shape (0=Round, 1=NonRound), Taste (0=NonSweet,
1=Sweet). The export_text output shows the actual tree that CART built — it's a human-
readable flowchart you can follow manually. We then test it on three new fruit
descriptions to check predictions. Scikit-learn trees need numbers, which is why the
categories are encoded as integers first. Figure 14.1 draws this same tree. Notice that
it never asks about Shape: color, size and taste are enough to tell these twelve fruits
apart.

Run it:           python ex2_cart_decision_tree_with_scikit_learn_fruit_cl.py
Expected output:  expected_output/ex2_cart_decision_tree_with_scikit_learn_fruit_cl.txt
"""
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import numpy as np

# Encoding: Color:0=Green,1=Yellow,2=Red | Size:0=Small,1=Med,2=Big
# Shape:0=Round,1=NonRound | Taste:0=NonSweet,1=Sweet
X = np.array([
    [0,2,0,0],[0,2,0,0],             # Watermelon
    [2,1,0,1],[0,1,0,0],[2,1,0,1],  # Apple
    [1,1,0,0],[1,2,0,0],             # Lemon, Grapefruit
    [1,0,1,0],[1,0,1,0],             # Banana
    [2,0,0,1],                        # Cherry
    [2,0,0,0],[2,0,1,0],             # Grape
])
y = np.array([6,6, 0,0,0, 5,4, 1,1, 2, 3,3])
class_names   = ['Apple','Banana','Cherry','Grape','Grapefruit','Lemon','Watermelon']
feature_names = ['Color','Size','Shape','Taste']

clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
clf.fit(X, y)

print("=== Decision Tree (CART, criterion=entropy) ===\n")
print(export_text(clf, feature_names=feature_names, class_names=class_names))

print(f"Training Accuracy: {accuracy_score(y, clf.predict(X))*100:.1f}%\n")

print("=== Predictions on New Fruit Descriptions ===")
tests = {
    'Red, Small, Round, Sweet'    : [2, 0, 0, 1],
    'Yellow, Big, Round, NonSweet': [1, 2, 0, 0],
    'Green, Big, Round, NonSweet' : [0, 2, 0, 0],
}
for desc, feat in tests.items():
    pred = clf.predict([feat])[0]
    print(f"  {desc:<45} -> {class_names[pred]}")
