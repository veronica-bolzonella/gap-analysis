from bs4 import BeautifulSoup
import pandas as pd
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig


from src.config import DOMAIN_URL


def scrape_minors_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    result = []

    for h4 in soup.find_all("h4"):
        a_tag = h4.find("a")
        if a_tag and a_tag.has_attr("href"):
            url = a_tag["href"]
            name = a_tag.get_text(strip=True)
            result.append({"name": name, "url": DOMAIN_URL + url})

    return pd.DataFrame(result)


async def crawl_url(url):

    run_config = CrawlerRunConfig(
        css_selector="#content",  # Targets <main id="content" ...>
        cache_mode="BYPASS",  # Optional: always fetch fresh content
    )
    async with AsyncWebCrawler() as crawler:
        # Run the crawler on a URL
        result = await crawler.arun(url=url, config=run_config)

        # Print the extracted content
        return result.markdown
