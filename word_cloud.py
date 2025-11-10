import matplotlib.pyplot as plt
from wordcloud import WordCloud

text = "Statistics, Mathematics, EDA, Visualization,Data Analysis, Categorical Data, Numerical Data, Data Science, Python, GitHub, dumping, Training, Testing,Deployment, cloud, correlation, feature engineering, cross_validation"
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
