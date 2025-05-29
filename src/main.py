import requests
import os
from bs4 import BeautifulSoup
import pandas as pd
import asyncio
from tqdm import tqdm


def clean_summary_output(response_text):
    # Remove <think>...</think> blocks if present
    import re

    return re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()


def chat_with_model(markdown_content):
    api_key = os.environ.get("NOLAI_API_KEY")
    url = "https://chat.nolai.fyi/api/chat/completions"
    model = "deepseek-r1:8b"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes academic program descriptions.",
            },
            {
                "role": "user",
                "content": f"Please summarize the following program description in two lines. In your summary: The main learning objectives or skills will develop and the key technologies or domain involved:\n\n{markdown_content}",
            },
        ],
    }

    response = requests.post(url, headers=headers, json=data)
    text_response = response.json()["choices"][0]["message"]["content"]
    clean_text_response = clean_summary_output(text_response)
    return clean_text_response


def summarize_minors(df, checkpoint_path="minors_checkpoint.csv"):
    # Load existing progress if available
    if os.path.exists(checkpoint_path):
        df_checkpoint = pd.read_csv(checkpoint_path)
        df_valid = df_checkpoint[
            df_checkpoint["summary"].notna()
            & (df_checkpoint["summary"].str.strip() != "")
        ]
        processed_urls = set(df_valid["url"].dropna())
    else:
        df_checkpoint = df.copy()
        df_checkpoint["markdown"] = ""
        df_checkpoint["summary"] = ""
        processed_urls = set()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing Minors"):
        url = row["url"]
        if url in processed_urls:
            print(f"done: {i}")

            continue  # already done

        try:
            # Crawl
            markdown = asyncio.run(crawl_url(url))

            # Summarize
            summary = chat_with_model(markdown)

            # Store results
            df_checkpoint.loc[i, "markdown"] = markdown
            df_checkpoint.loc[i, "summary"] = summary

            # Save after each successful item
            df_checkpoint.to_csv(checkpoint_path, index=False)
            print(f"Saved row {i}: {row['name']}")

        except Exception as e:
            print(f"Error at row {i} ({url}): {e}")
            continue

    # Final save
    df_checkpoint.to_csv("minors_with_markdown_and_summaries.csv", index=False)


def main():
    df = scrape_minors_from_file("minors.html")

    summarize_minors(df)


if __name__ == "__main__":
    main()
