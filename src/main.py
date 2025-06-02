import requests
import os
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import asyncio
from tqdm import tqdm

from src.config import DATA_DIR

from src.webscraper import HTMLScrapper
import chardet


def main():

    # Your enrichment data
    data = [
        ["3D Solidworks", "Geen", 0, 0],
        ["Aan de slag met CSRD: rapporteren over duurzaamheid", "Data Science", 1, 1],
        ["Adviseur Jeugdopleiding Sport & Bewegen", "Geen", 0, 0],
        ["AI en Recht - Toepassingen in de praktijk", "AI", 2, 3],
        ["Data Science in de praktijk", "Data Science", 2, 3],
        ["DNA Sequencing - NGS & TGS", "Geen", 0, 0],
        ["Engels: Academic Writing", "Geen", 0, 0],
        ["Fermentatietechnologie in de biobased economy (praktijk)", "Geen", 0, 0],
        ["ISAK level 1 – Meten van lichaamssamenstelling", "Geen", 0, 0],
        ["ISAK level 2 – Meten van lichaamssamenstelling", "Geen", 0, 0],
        ["Klinische cytologie", "Geen", 0, 0],
        [
            "Leefstijlcoaching voor GGZ-verpleegkundigen en Verpleegkundig Specialisten",
            "Geen",
            0,
            0,
        ],
        ["Nederlands A1 & A2 (avond)", "Geen", 0, 0],
        ["NL-Actief branchediploma’s Fitness", "Geen", 0, 0],
        ["Podcast maken en lanceren", "Geen", 0, 0],
        ["Social Media (gevorderden)", "AI", 1, 2],
        ["Wiskunde (zomeropfriscursus)", "Geen", 0, 0],
        ["Zakelijk Duits", "Geen", 0, 0],
    ]

    # Load your CSV data
    df1 = pd.read_csv(DATA_DIR / "courses" / "course_descriptions.csv")
    df2 = pd.read_csv(
        DATA_DIR / "courses" / "courses_full_table_laurie.csv",
        sep=";",
        encoding="ISO-8859-1",
        quotechar='"',
        engine="python",
        on_bad_lines="warn",
    )

    # Convert your `data` to a DataFrame
    enrichment_df = pd.DataFrame(
        data,
        columns=[
            "Naam opleiding",
            "Sleuteltechnologie",
            "Sleuteltechnologiecategorie (0–3)",
            "Mate van technologie (0–3)",
        ],
    )

    # Ensure enrichment_df matches df2's columns
    for col in df2.columns:
        if col not in enrichment_df.columns:
            enrichment_df[col] = pd.NA

    for col in enrichment_df.columns:
        if col not in df2.columns:
            df2[col] = pd.NA

    # Reorder columns before appending
    enrichment_df = enrichment_df[df2.columns]

    # ✅ Append the rows to df2
    df2 = pd.concat([df2, enrichment_df], ignore_index=True)

    # Merge with df1 using 'name' and 'Naam opleiding'
    merged = pd.merge(
        df1, df2, left_on="name", right_on="Naam opleiding", how="left", indicator=True
    )

    # Print merge statistics
    match_counts = merged["_merge"].value_counts()
    print("\nMerge results:")
    print(match_counts)
    print(f"\n✅ Rows merged correctly: {match_counts.get('both', 0)}")
    print(f"❌ Rows with no match in df2: {match_counts.get('left_only', 0)}")

    # Save merged file
    merged.to_csv(DATA_DIR / "courses" / "merged.csv", index=False)

    # Save unmatched course names (optional)
    unmatched = merged[merged["_merge"] == "left_only"]
    if not unmatched.empty:
        unmatched_names = unmatched[["name"]]
        unmatched_names.to_csv(DATA_DIR / "courses" / "unmatched.csv", index=False)
        print(f"\n⚠️ Unmatched rows saved to 'unmatched.csv'")


if __name__ == "__main__":
    main()
