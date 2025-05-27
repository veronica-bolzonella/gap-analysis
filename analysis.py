from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from fpdf import FPDF
import pandas as pd
import numpy as np
import re
import matplotlib.pyplt as plt

def load_and_preprocess_data(filepath, ai_trends_text):
    # Load CSV
    df_courses = pd.read_csv(filepath)

    # Translate 'Toelichting' column if 'summary' doesn't exist
    if "summary" not in df_courses.columns:
        translator = GoogleTranslator(source='nl', target='en')
        df_courses["description"] = df_courses["Toelichting"].astype(str).apply(
            lambda text: translator.translate(text) if pd.notnull(text) else ""
        )
        df_courses.rename(columns={"description": "summary"}, inplace=True)

    # Clean up the text
    def preprocess(text):
        text = re.sub(r"\*\*.*?\*\*", "", text)  # remove markdown bold
        text = re.sub(r"\s+", " ", text)  # normalize whitespace
        return text.lower()

    df_courses["clean"] = df_courses["summary"].astype(str).apply(preprocess)

    # Filter out courses with Sleuteltechnologiecategorie == 0 if the column exists
    if "Sleuteltechnologiecategorie (0-3)" in df_courses.columns:
        df_courses = df_courses[df_courses["Sleuteltechnologiecategorie (0–3)"] != 0]

    # Ensure 'name' column exists
    if "name" not in df_courses.columns:
        df_courses.rename(columns={"Naam opleiding": "name"}, inplace=True)

    # Prepare AI trend list
    ai_trend_list = [trend.strip() for trend in ai_trends_text.strip().split(",")]
    ai_trend_list_clean = [preprocess(trend) for trend in ai_trend_list]

    return df_courses, ai_trend_list, ai_trend_list_clean

def compute_similarity(df_courses, ai_trend_list_clean, coverage_threshold=0.3):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode AI trends and course descriptions
    trend_embeddings = model.encode(ai_trend_list_clean, convert_to_tensor=True)
    course_embeddings = model.encode(df_courses["clean"].tolist(), convert_to_tensor=True)

    # Compute similarity matrix
    similarity_matrix = util.cos_sim(course_embeddings, trend_embeddings).cpu().numpy()
    df_courses["ai_trend_similarity"] = similarity_matrix.max(axis=1)

    # Determine covered trends per course
    covered_trends_per_course = []
    for course_similarities in similarity_matrix:
        covered_trends = [ai_trend_list_clean[i] for i, score in enumerate(course_similarities) if score >= coverage_threshold]
        covered_trends_per_course.append(covered_trends)

    df_courses["covered_trends"] = covered_trends_per_course

    # Generate trend coverage matrix
    trend_coverage_matrix = pd.DataFrame(similarity_matrix >= coverage_threshold,
                                         columns=ai_trend_list_clean,
                                         index=df_courses["name"])
    return df_courses, trend_coverage_matrix, similarity_matrix

def generate_report(df_courses, trend_coverage_matrix, ai_trend_list_clean, coverage_threshold=0.3):
    trend_coverage_matrix.to_csv("trend_coverage_matrix.csv")

    top_matches = df_courses.sort_values(by="ai_trend_similarity", ascending=False).head(10)
    covered_trends = set([trend for trends in df_courses["covered_trends"] for trend in trends])
    uncovered_trends = set(ai_trend_list_clean) - covered_trends

    covered_courses = df_courses[df_courses["ai_trend_similarity"] >= coverage_threshold]
    coverage_rate = len(covered_courses) / len(df_courses)
    average_similarity = df_courses["ai_trend_similarity"].mean()
    max_similarity = df_courses["ai_trend_similarity"].max()
    min_similarity = df_courses["ai_trend_similarity"].min()

    # === Save histogram ===
    plt.figure(figsize=(12, 6))
    plt.hist(df_courses["ai_trend_similarity"], bins=30, color="skyblue", edgecolor="black")
    plt.axvline(coverage_threshold, color="red", linestyle="--", label=f"Coverage threshold ({coverage_threshold})")
    plt.title("Distribution of AI Trend Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Number of Courses")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("histogram_embeddings.jpeg")
    plt.close()

    # === Generate report text ===
    report = f"""
            AI Trends Coverage Report (Based on {len(df_courses)} Courses with Sleuteltechnologiecategorie != 0)

            1. Similarity Statistics
            - Coverage threshold: {coverage_threshold}
            - Average similarity score: {average_similarity:.3f}
            - Maximum similarity score: {max_similarity:.3f}
            - Minimum similarity score: {min_similarity:.3f}

            2. Interpretation
            - ~{coverage_rate:.0%} of the courses show at least some textual alignment with emerging AI trends.
            - The remaining ~{(1 - coverage_rate):.0%} may not explicitly address AI, Data Science, or their societal/technical implications.
            - High scoring courses tend to focus on applied AI (e.g. computer vision, content generation, smart industry).
            - Low scoring courses may still touch relevant themes, but not in explicitly AI-framed language.

            3. Recommendations
            - Consider integrating AI-relevant themes (ethics, tools, applications) into low-alignment courses.
            - Use high-scoring courses as models for curriculum innovation.
            - A thematic breakdown per trend (e.g. sustainability vs education vs tools) would allow deeper curriculum strategy.

            4. Details
            - Top 10 Courses Matching AI Trends:
            {top_matches['name'].to_string(index=False)}

            Trends Not Covered:
            - {', '.join(sorted(uncovered_trends))}

            Methodology
            - Excluded courses with no relation to the topic (i.e. Dutch A1).
            - Used contextual embeddings (SentenceTransformer: 'all-MiniLM-L6-v2') for semantic similarity.
            - Trend coverage determined by cosine similarity ≥ {coverage_threshold} for a given trend.
            - Trend coverage matrix saved as 'trend_coverage_matrix.csv'.
        """
    print(report)

    # === Generate PDF ===
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "AI Trends Coverage Report", ln=True, align="C")
            self.ln(10)
        def chapter_body(self, text):
            self.set_font("Arial", "", 11)
            self.multi_cell(0, 10, text)
            self.ln()

    pdf = PDF()
    pdf.add_page()
    clean_report = report.encode("latin-1", "replace").decode("latin-1")
    pdf.chapter_body(clean_report)
    pdf.output("AI_Trends_Coverage_Report.pdf")

def main(filepath, ai_trends_text, coverage_threshold):
    df_courses, ai_trend_list, ai_trend_list_clean = load_and_preprocess_data(filepath, ai_trends_text)
    df_courses, trend_coverage_matrix, similarity_matrix = compute_similarity(df_courses, ai_trend_list_clean, coverage_threshold)
    generate_report(df_courses, trend_coverage_matrix, ai_trend_list, coverage_threshold)

if __name__ == "__main__":
    filepath = "minors.csv"
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
    coverage_threshold = 0.3
    main(filepath, ai_trends_text, coverage_threshold)
