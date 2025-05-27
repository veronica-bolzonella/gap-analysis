from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from fpdf import FPDF
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util


# === Step 1: Load and preprocess ===

# Define AI trends reference (from PDF summary)
ai_trends_text = """
AI supply chain, sustainability, ethics, bias, digital colonialism, surveillance, privacy,
student-focused AI tools, plagiarism detection,
AI-generated learning materials, Learning Management Systems, scheduling optimization,
healthcare diagnostics, predictive analytics, fraud detection, robo-advisors, Large Language Models, Natural Language Processing,
machine learning
computer vision, deep learning, chatbots, content generation, autonomous vehicles, smart mobility,
generative AI for media, AI in manufacturing, cobots, predictive maintenance, digital twins,
personalized recommendations, immersive media, Explainable AI, algorithmic decision-making, robotics,.
"""

# Load the new Dutch CSV with course data
df_courses = pd.read_csv("minors.csv")

# Translate the 'Toelichting' column from Dutch to English
if "summary" not in df_courses.columns:
    translator = GoogleTranslator(source='nl', target='en')
    df_courses["description"] = df_courses["Toelichting"].astype(str).apply(
      lambda text: translator.translate(text) if pd.notnull(text) else ""
  )

# Clean up the text
def preprocess(text):
    text = re.sub(r"\*\*.*?\*\*", "", text)  # remove markdown bold
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.lower()


# Filter out courses with Sleuteltechnologiecategorie == 0
if "Sleuteltechnologiecategorie (0-3)" in df_courses.columns:
  df_courses = df_courses[df_courses["Sleuteltechnologiecategorie (0–3)"] != 0]

# Prepare trend list
ai_trend_list = [trend.strip() for trend in ai_trends_text.strip().split(",")]
ai_trend_list_clean = [preprocess(trend) for trend in ai_trend_list]

df_courses["clean"] = df_courses["summary"].astype(str).apply(preprocess)

if "name" not in df_courses.columns:
    df_courses.rename(columns={"Naam opleiding": "name"}, inplace=True)
    
    
    
###

# ANALYSIS

###

# === Compute embeddings using SentenceTransformer ===
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode AI trends and course descriptions
trend_embeddings = model.encode(ai_trend_list_clean, convert_to_tensor=True)
course_embeddings = model.encode(df_courses["clean"].tolist(), convert_to_tensor=True)

# Compute course-trend similarity matrix
similarity_matrix = util.cos_sim(course_embeddings, trend_embeddings).cpu().numpy()

# Add max similarity (best matching trend) per course
max_similarities = similarity_matrix.max(axis=1)
df_courses["ai_trend_similarity"] = max_similarities

# Determine which trends are covered by each course
coverage_threshold = 0.3  # Tunable threshold for 'covering' a trend
covered_trends_per_course = []
for course_similarities in similarity_matrix:
    covered_trends = [ai_trend_list[i] for i, score in enumerate(course_similarities) if score >= coverage_threshold]
    covered_trends_per_course.append(covered_trends)

df_courses["covered_trends"] = covered_trends_per_course

# === Generate trend coverage matrix ===
trend_coverage_matrix = pd.DataFrame(similarity_matrix >= coverage_threshold,
                                     columns=ai_trend_list,
                                     index=df_courses["name"])

# Save matrix to CSV
trend_coverage_matrix.to_csv("trend_coverage_matrix.csv")


# Results summary
top_matches = df_courses.sort_values(by="ai_trend_similarity", ascending=False).head(10)
covered_trends = set([trend for trends in df_courses["covered_trends"] for trend in trends])
uncovered_trends = set(ai_trend_list) - covered_trends

covered_courses = df_courses[df_courses["ai_trend_similarity"] >= coverage_threshold]
not_covered_courses = df_courses[df_courses["ai_trend_similarity"] < coverage_threshold]

coverage_rate = len(covered_courses) / len(df_courses)
average_similarity = df_courses["ai_trend_similarity"].mean()
max_similarity = df_courses["ai_trend_similarity"].max()
min_similarity = df_courses["ai_trend_similarity"].min()


print(f"Average similarity score: {average_similarity:.3f}")
print(f"Maximum similarity score: {max_similarity:.3f}")
print(f"Minimum similarity score: {min_similarity:.3f}")
print(f"Coverage threshold: {coverage_threshold}")

print(f"\nTop 10 Courses Matching AI Trends:")
print(top_matches["name"].to_string(index=False))

print("\nTrends Not Covered:")
print(", ".join(sorted(uncovered_trends)))


print("\nTrends Covered:")
print(", ".join(sorted(covered_trends)))