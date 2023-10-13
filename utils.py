import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
import random
from bs4.element import Comment
from bs4 import BeautifulSoup
from urllib.parse import quote
import readline
import logging


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


log_handler = logging.FileHandler("data/log.log")
log_handler.setLevel(logging.DEBUG)
log_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(log_handler)
    return logger


options = FirefoxOptions()
options.add_argument("--headless")
options.binary = "/usr/bin/firefox-esr"
options.set_preference("general.useragent.override",
                       "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0")
browser = webdriver.Firefox(options)
browser.set_page_load_timeout(10)


def search(query: str, max_num=3):
    search_url = f"https://www.bing.com/search?q={quote(query)}"
    try:
        browser.get(search_url)
        time.sleep(1)
        soup = BeautifulSoup(browser.page_source, "lxml")

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
        raise RuntimeError("api error")

    # read the first 1 in detail
    if len(results) >= 1:
        try:
            browser.get(results[0][2])
            time.sleep(1)
            soup = BeautifulSoup(browser.page_source, "lxml")
            texts = []
            total_len = 0
            for text in soup.find_all(string=True):
                if total_len > 600:
                    break
                if text.parent.name in ["script", "style", "head", "title", "meta", "[document]"]:
                    continue
                if isinstance(text, Comment):
                    continue
                text = text.strip()
                if len(text) < 10:
                    continue
                texts.append(text)
                total_len += len(text)
            results[0] = (results[0][0], "\n".join(
                (results[0][1], *texts)), results[0][2])
        # selenium timeout
        except TimeoutError:
            pass

    return results[:max_num]


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
