import feedparser
import requests
from bs4 import BeautifulSoup
from readability import Document

def get_full_article(url):
    try:
        r = requests.get(url, timeout=5)
        doc = Document(r.text)
        html = doc.summary()

        soup = BeautifulSoup(html, "html.parser")
        return " ".join([p.get_text() for p in soup.find_all("p")])
    except:
        return ""

def get_news(source="BBC"):
    feeds = {
        "BBC": "https://feeds.bbci.co.uk/turkce/rss.xml",
        "CNN": "http://rss.cnn.com/rss/edition.rss",
        "TRT": "https://www.trthaber.com/sondakika.rss"
    }

    feed = feedparser.parse(feeds.get(source))
    articles = []

    for e in feed.entries[:5]:
        text = get_full_article(e.link)

        if len(text) < 200:
            text = e.title + " " + e.get("description", "")

        articles.append({"title": e.title, "text": text})

    return articles