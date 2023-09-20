import requests
from bs4 import BeautifulSoup


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def search(query: str):
    base_url = "https://cn.bing.com/search?q="
    search_url = base_url + query

    try:
        response = requests.get(search_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for result in soup.find_all("li", class_="b_algo"):
            title = result.find("h2").get_text()
            content = result.find("p").get_text()
            link = result.find("a")["href"]
            results.append((title, content, link))

        return results

    except Exception as e:
        print(f"Error: {e}")
        return []

