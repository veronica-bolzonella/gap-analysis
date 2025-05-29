from bs4 import BeautifulSoup
import pandas as pd
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
import asyncio
from tqdm import tqdm
from src.config import DOMAIN_URL, DATA_DIR
from pathlib import Path


class HTMLScrapper:
    @staticmethod
    def scrape_minors(file_path):
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

    @staticmethod
    def scrape_courses(file_name):
        """
        Extracts all <a class="finder-result__title"> tags from an HTML file
        and returns a DataFrame with 'name' and full 'url'.
        """
        file_path = Path.joinpath(DATA_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")
        results = []

        for a in soup.find_all("a", class_="finder-result__title"):
            name = a.get_text(strip=True)
            relative_url = a.get("href", "")
            full_url = DOMAIN_URL + relative_url
            results.append({"name": name, "url": full_url})

        return pd.DataFrame(results)


async def crawl_url(url, semaphore):

    async with semaphore:
        run_config = CrawlerRunConfig(cache_mode="BYPASS")
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=run_config)
            return result.markdown


async def crawl_all_urls(urls, thread_num=30):
    semaphore = asyncio.Semaphore(thread_num)

    tasks = [crawl_url(url, semaphore) for url in urls]
    results = []

    for coro in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Crawling URLs"
    ):
        result = await coro
        results.append(result)

    return results


def main():
    df = HTMLScrapper.scrape_courses("courses.html")

    urls = df["url"].tolist()

    print(f"Crawling {len(df)} URLs")

    markdown_results = asyncio.run(crawl_all_urls(urls, thread_num=25))
    df["markdown"] = markdown_results

    # Preview and optionally save
    print(df[["name", "url", "markdown"]].head())
    df.to_csv(Path.joinpath(DATA_DIR, "course_descriptions.csv"), index=False)
    print("Saved results to courses_descriptions.csv")


if __name__ == "__main__":
    main()
