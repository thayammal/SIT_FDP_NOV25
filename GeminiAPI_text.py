import os
import pandas as pd
#from google import generativeai as genai   # ensure you have the right library
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



from google import genai
client = genai.Client(api_key="AIzaSyAtWKx71xMKbXymVEEdL6xKTCfe3ZAYV-4")

# 1. Configure your API key
#genai.configure(api_key=os.getenv("AIzaSyAtWKx71xMKbXymVEEdL6xKTCfe3ZAYV-4"))

# 2. Load your data
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
        "It would be great if the mobile app supported offline mode.",
        "Please consider adding a dark mode option in the next update.",
        "I suggest integrating with the Google Drive platform.",
        "The ability to export data as a PDF is missing and would be very useful.",
        "Can you make the dashboard reports customizable?",
        "The website is crashing every time I try to save my settings.",
        "I received an error 404 when clicking the main link.",
        "The payment processor seems to be completely broken.",
        "The chat feature isn't sending messages reliably after the last patch.",
        "I noticed a typo in the main header of your contact page.",
    ],
    'category': [
        "Positive_Review", "Negative_Review", "Support_Query", "Positive_Review",
        "Negative_Review", "Support_Query", "Product_Pricing", "Product_Pricing",
        "Product_Pricing", "Product_Pricing", "Product_Pricing",
        "Feature_Request", "Feature_Request", "Feature_Request", "Feature_Request",
        "Feature_Request", "Bug_Report", "Bug_Report", "Bug_Report", "Bug_Report",
        "Bug_Report",
    ]
}
df = pd.DataFrame(data)

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['category'], test_size=0.3, random_state=42, stratify=df['category']
)


def gemini_classify(text, categories):
    prompt = f"Classify the following text into one of these categories: {', '.join(categories)}. Text: \"{text}\" Category:"
    response = client.models.generate_content(
        model="gemini-2.5-flash",   # use correct model name
        contents=prompt
    )
    return response.text.strip()

# # 4. Function to call Gemini for classification
# def gemini_classify(text, categories):
#     prompt = f"""
#     Classify the following text into one of these categories: {', '.join(categories)}.
#     Text: \"{text}\"
#     Category:
#     """
#     response = genai.models.generate_content(
#         model="gemini-2.0-flash",   # choose the correct model version
#         contents=[{"parts":[{"text": prompt}]}]
#     )
#     # The result text may include the predicted category
#     result = response.text.strip()
#     return result

# 5. Use Gemini on test set and evaluate
categories = df['category'].unique().tolist()
y_pred = []
for text in X_test:
    pred = gemini_classify(text, categories)
    y_pred.append(pred)

print(classification_report(y_test, y_pred, zero_division=0))

new_sentences = [
    "I am having trouble logging into my account and need help immediately.",
    "Could you provide details about the enterprise pricing plan and setup fee?",
    "It would be awesome if you added a dark mode and offline support in the next version."
]

for s in new_sentences:
    pred = gemini_classify(s, categories)
    print(f"Text: {s}\nPredicted Category: {pred}\n")
    