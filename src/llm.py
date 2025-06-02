import os
from pathlib import Path
import requests
from tqdm import tqdm


def clean_summary_output(response_text):
    # Remove <think>...</think> blocks if present
    import re

    return re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()


def chat_with_model(markdown_content):
    api_key = os.environ.get("NOLAI_API_KEY")
    url = "https://chat.nolai.fyi/api/chat/completions"
    model = "gemma3:4b"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that summarizes academic course descriptions. "
                    "Your task is to read and analyze a course description provided in markdown format. "
                    "From this content, extract and summarize the core learning objectives and the key skills students will develop. "
                    "Also highlight the relevant technologies or domains the course focuses on. "
                    "Return a short, clear summary (2-3 sentences) suitable for students evaluating the course."
                ),
            },
            {"role": "user", "content": markdown_content},
        ],
    }

    response = requests.post(url, headers=headers, json=data)
    text_response = response.json()["choices"][0]["message"]["content"]
    clean_text_response = clean_summary_output(text_response)
    return clean_text_response


def summarize_minors(df, checkpoint_path: Path, result_path: Path):
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

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing"):
        url = row["url"]
        if url in processed_urls:
            print(f"done: {i}")

            continue  # already done

        try:
            # Crawl
            markdown = row["markdown"]
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
    df_checkpoint.to_csv(result_path, index=False)
