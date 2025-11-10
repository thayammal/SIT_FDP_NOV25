import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# --- 1. Sample Data (Replace with your actual data loading) ---
# In a real scenario, you'd load a dataset from a file (e.g., CSV).
data = {
    'text': [
        "The new product launch was a huge success, everyone loves it!",
        "This service is slow and completely unreliable, what a disappointment.",
        "I need help with my account settings and billing information.",
        "Absolutely fantastic quality and fast delivery, highly recommend.",
        "The software crashed multiple times, resulting in data loss. Terrible experience.",
        "General inquiry about company policies and business hours.",
        "What are the monthly fees for your premium plan?",
        "Can you send me the full price list for all subscription tiers?",
        "Is there a one-time setup charge for the new software?",
        "I'm confused about the latest invoice—please clarify the charges.",
        "Do you offer any discounts for yearly contracts?",
        # Feature_Request
        "It would be great if the mobile app supported offline mode.",
        "Please consider adding a dark mode option in the next update.",
        "I suggest integrating with the Google Drive platform.",
        "The ability to export data as a PDF is missing and would be very useful.",
        "Can you make the dashboard reports customizable?",
        # Bug_Report
        "The website is crashing every time I try to save my settings.",
        "I received an error 404 when clicking the main link.",
        "The payment processor seems to be completely broken.",
        "The chat feature isn't sending messages reliably after the last patch.",
        "I noticed a typo in the main header of your contact page.",
    ],
    'category': [
        "Positive_Review",
        "Negative_Review",
        "Support_Query",
        "Positive_Review",
        "Negative_Review",
        "Support_Query",
        "Product_Pricing", 
        "Product_Pricing", 
        "Product_Pricing", 
        "Product_Pricing", 
        "Product_Pricing",
        "Feature_Request", 
        "Feature_Request", 
        "Feature_Request", 
        "Feature_Request",
        "Feature_Request",
        "Bug_Report", 
        "Bug_Report", 
        "Bug_Report", 
        "Bug_Report", 
        "Bug_Report",
    ]
}
df = pd.DataFrame(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['category'], 
    test_size=0.3, 
    random_state=42
)

# --- 2. Define the Pipeline ---
# A pipeline sequentially applies a list of transformers and a final estimator (model).
text_clf_pipeline = Pipeline([
    # Step 1: Feature Extraction/Transformation
    # TfidfVectorizer converts text into a matrix of TF-IDF features.
    ('tfidf', TfidfVectorizer(stop_words='english')),
    
    # Step 2: Classifier (The AI Model)
    # Multinomial Naive Bayes is a common baseline for text classification.
    ('clf', MultinomialNB()),
])

# --- 3. Train the Model ---
text_clf_pipeline.fit(X_train, y_train)

# --- 4. Evaluate the Model ---
# Use the trained pipeline to predict on the test set
y_pred = text_clf_pipeline.predict(X_test)


# --- 5. Make a Prediction on New Data ---
new_text = ["Have you considered adding multi-language support to the app?"]
prediction = text_clf_pipeline.predict(new_text)

print(f"\nNew Text: '{new_text[0]}'")
print(f"Predicted Category: {prediction[0]}")