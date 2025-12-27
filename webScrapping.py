## importing required package


import requests
import pandas as pd
from bs4 import BeautifulSoup

## Request the page


url = "https://quotes.toscrape.com/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

#Extract titles
#title in this page use tag h1 
titles = [soup.find("h1").get_text(strip=True)]



## Extracting titles



#title in this page use tag h1 
titles = [soup.find("h1").get_text(strip=True)]


print(titles)

## extracting quotes and authors of each quote

quote_blocks = soup.find_all("div", class_="quote")

quotes = []
authors = []

for block in soup.find_all("div", class_="quote"):
    quote = block.find("span", class_="text").get_text(strip=True)
    author = block.find("small", class_="author").get_text(strip=True)
    
    
    quotes.append(quote)
    authors.append(author)

print(quotes)
print(authors)

## create dataFrame


texts_col = titles + quotes
types_col = ["Title"] * len(titles) + ["Quote"] * len(quotes)
author_col = [None] * len(titles) + authors
df = pd.DataFrame({
    "Text": texts_col,
    "Type": types_col,
    "Author": author_col
})

df.head() #show the first lines of dataframe

## add word count to dataframe


df["Word_Count"] = df["Text"].apply(lambda x: len(x.split()))

df.head()

#  Visualization


import matplotlib.pyplot as plt

# Keep only quotes
quotes_df = df[df["Type"] == "Quote"]

plt.figure(figsize=(8,5))
plt.hist(quotes_df["Word_Count"], bins=5)

plt.xlabel("Word Count per Quote")
plt.ylabel("Number of Quotes")
plt.title("Distribution of Quote Lengths")
plt.show()



## extract the more 5 frequents words for each author 


#### we will use mapereduce method from the last chapter 



#the same mapper as the last homework with input text and author
import re

def mapp(text_input, author):
    """
    Maps text into ((author, word), 1) tuples
    """
    mapped = []
    
    # Ensure input is string and lowercase
    text = str(text_input).lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split into words
    words = text.split()
    
    # Create tuples for each word
    for word in words:
        mapped.append(((author, word), 1))
        
    return mapped
    
mapped_all = []

for _, row in quotes_df.iterrows():
    author = row["Author"]
    text = row["Text"]
    
    mapped_all.extend(mapp(text, author))

print(mapped_all[:20])  # preview first 10 mapped tuples


from collections import defaultdict

#shuffle
reduced = defaultdict(list)
for key, value in mapped_all:
    reduced[key].append(value)

#reduce sum counts for each (author, word)
reduced_counts = {key: sum(values) for key, values in reduced.items()}

#words per author
author_word_freq = defaultdict(list)
for (author, word), count in reduced_counts.items():
    author_word_freq[author].append((word, count))

#Sort top 5 words per author
for author in author_word_freq:
    author_word_freq[author] = sorted(author_word_freq[author], key=lambda x: x[1], reverse=True)[:5]




## Display results

    


# 4. Display results
for author, top_words in author_word_freq.items():
    print(f"{author}: {top_words}")
    


## visualization 

    


author = 'Albert Einstein' 
top_words = author_word_freq[author]

words = [w for w, c in top_words]
counts = [c for w, c in top_words]

plt.figure(figsize=(8,5))
plt.bar(words, counts, color='pink')
plt.title(f"Top words for {author}")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()

