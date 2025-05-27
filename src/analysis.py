from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from fpdf import FPDF

# === Step 1: Load and preprocess ===

# Define AI trends reference (from PDF summary)
ai_trends_text = """
AI supply chain, sustainability, ethics, bias, digital colonialism, surveillance, privacy, 
student-focused AI tools (Duolingo, Grammarly, Preply), AI tutors, plagiarism detection, 
AI-generated learning materials, Learning Management Systems, scheduling optimization, 
healthcare diagnostics, predictive analytics, fraud detection, robo-advisors, LLMs, NLP, 
computer vision, deep learning, chatbots, content generation, autonomous vehicles, smart mobility, 
generative AI for media, AI in manufacturing (cobots, predictive maintenance), digital twins, 
personalized recommendations, immersive media, Explainable AI, algorithmic decision-making.
"""

# Load the CSV with course summaries
df_courses = pd.read_csv("minors_checkpoint.csv")


# Clean up the text
def preprocess(text):
    text = re.sub(r"\*\*.*?\*\*", "", text)  # remove markdown bold
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.lower()


df_courses["summary_clean"] = df_courses["markdown"].astype(str).apply(preprocess)
ai_trends_clean = preprocess(ai_trends_text)

# === Step 2: Compute similarity using TF-IDF ===
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(
    [ai_trends_clean] + df_courses["summary_clean"].tolist()
)
similarity_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
df_courses["ai_trend_similarity"] = similarity_scores

# === Step 3: Print top 15 course matches ===
top_matches = df_courses.sort_values(by="ai_trend_similarity", ascending=False).head(15)
print("\nTop Courses Matching AI Trends:\n")
print(top_matches[["name", "ai_trend_similarity"]])

# === Step 4: Plot histogram of scores ===
plt.figure(figsize=(12, 6))
plt.hist(similarity_scores, bins=30, color="skyblue", edgecolor="black")
plt.axvline(0.08, color="red", linestyle="--", label="Coverage threshold (0.08)")
plt.title("Distribution of AI Trend Similarity Scores Across Course Summaries")
plt.xlabel("Similarity Score")
plt.ylabel("Number of Courses")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("histogram.jpeg")

# === Step 5: Generate and print summary report ===
coverage_threshold = 0.08
covered_courses = df_courses[df_courses["ai_trend_similarity"] >= coverage_threshold]
not_covered_courses = df_courses[df_courses["ai_trend_similarity"] < coverage_threshold]

coverage_rate = len(covered_courses) / len(df_courses)
average_similarity = df_courses["ai_trend_similarity"].mean()
max_similarity = df_courses["ai_trend_similarity"].max()
min_similarity = df_courses["ai_trend_similarity"].min()

report = f"""
🧠 AI Trends Coverage Report (Based on {len(df_courses)} Course Summaries)

1. Overall Coverage
- Courses touching AI trends (similarity ≥ {coverage_threshold}): {len(covered_courses)} ({coverage_rate:.0%})
- Courses not aligned with AI trends: {len(not_covered_courses)} ({(1 - coverage_rate):.0%})

2. Similarity Statistics
- Average similarity score: {average_similarity:.3f}
- Maximum similarity score: {max_similarity:.3f}
- Minimum similarity score: {min_similarity:.3f}

3. Interpretation
- ~{coverage_rate:.0%} of HAN's minors show at least some textual alignment with emerging AI trends.
- The remaining ~{(1 - coverage_rate):.0%} may not explicitly address AI, Data Science, or their societal/technical implications.
- High scoring minors tend to focus on applied AI (e.g. computer vision, content generation, smart industry).
- Low scoring minors may still touch relevant themes, but not in explicitly AI-framed language.

4. Recommendations
- Consider integrating AI-relevant themes (ethics, tools, applications) into low-alignment minors.
- Use high-scoring minors as models for curriculum innovation.
- A thematic breakdown per trend (e.g. sustainability vs education vs tools) would allow deeper curriculum strategy.
"""

print(report)


# Create a simple PDF class
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "AI Trends Coverage Report", ln=True, align="C")
        self.ln(10)

    def chapter_body(self, text):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 10, text)
        self.ln()


# Initialize the PDF
pdf = PDF()
pdf.add_page()

clean_report = report.encode("latin-1", "replace").decode("latin-1")
pdf.chapter_body(clean_report)

# Save to file
pdf.output("ai_trends_coverage_report.pdf")

print("✅ Report saved to 'ai_trends_coverage_report.pdf'")

# Drop the helper column 'summary_clean' before saving
df_courses.drop(columns=["summary_clean"]).to_csv("gap_analysis.csv", index=False)
