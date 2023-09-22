from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
import random
from bs4.element import Comment
from bs4 import BeautifulSoup
from urllib.parse import quote
import readline


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


options = FirefoxOptions()
options.add_argument("--headless")
options.binary = "/usr/bin/firefox-esr"
options.set_preference("general.useragent.override", "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0")
browser = webdriver.Firefox(options)


def search(query: str, max_num=3):
    search_url = f"https://www.bing.com/search?q={quote(query)}"
    try:
        browser.get(search_url)
        soup = BeautifulSoup(browser.page_source, "html.parser")

        results = []
        for result in soup.find_all("li", class_="b_algo"):
            try:
                title = result.find("h2").get_text()
                link = result.find("a")["href"]
                abstract = result.find("p").get_text()
                results.append((title, abstract, link))
            except:
                pass
    except Exception as e:
        return []
    if len(results) == 0:
        return []

    title, abstract, link = random.choice(results[:5])
    browser.get(link)
    soup = BeautifulSoup(browser.page_source, "html.parser")
    texts = []
    total_len = 0
    for elem in soup.find_all(text=True):
        if total_len > 800:
            break
        if elem.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]'] or isinstance(elem, Comment):
            continue
        elem = elem.strip()
        if len(elem) < 10:
            continue
        texts.append(elem)
        total_len += len(elem)
    texts = abstract + '...\n' + ('\n'.join(texts))

    return title, texts, link


def get_news_mhy():
    return ()


def random_news():
    """Returns (time, title, content).
    """
    news_apis = [get_news_mhy]
    news = random.choice(news_apis)
    return news


if __name__ == '__main__':
    while True:
        inp = input(": ")
        print(search(inp))
