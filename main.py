import urllib.parse
import xml.etree.ElementTree as ET
import re
import spacy
import aiohttp
import asyncio
import time
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
from networkx import Graph
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Limit CPU usage
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Load spaCy model for preprocessing
nlp = spacy.load("en_core_web_sm")

import time

async def fetch_arxiv_data_with_retries(session, query, start=0, max_results=50, retries=3):
    query = urllib.parse.quote_plus(query)
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start={start}&max_results={max_results}'

    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
            if attempt == retries - 1:
                raise e



def extract_paper_details(xml_data):
    root = ET.fromstring(xml_data)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    papers = []
    for entry in root.findall('atom:entry', namespace):
        title = entry.find('atom:title', namespace).text
        summary = entry.find('atom:summary', namespace).text
        authors = [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)]
        doi = entry.find('atom:id', namespace).text if entry.find('atom:id', namespace) is not None else "N/A"
        papers.append({"title": title, "summary": summary, "authors": authors, "doi": doi})
    return papers

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_text_rank_with_bert(corpus):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus)
    similarity_matrix = cosine_similarity(embeddings)
    similarity_graph = Graph()

    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            if similarity_matrix[i][j] > 0.3:
                similarity_graph.add_edge(i, j, weight=similarity_matrix[i][j])

    text_rank_scores = nx.pagerank(similarity_graph, weight="weight")
    return text_rank_scores, similarity_graph


def calculate_k_truss_features(similarity_graph, k=3):
 
    try:
        k_truss_graph = nx.k_truss(similarity_graph, k=k)
        return k_truss_graph
    except Exception as e:
        print(f"Error calculating k-truss with k={k}: {e}")
        return None


def perform_clustering(corpus, num_clusters=5):
    """Performs K-Means clustering on the corpus."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), max_features=1000)
    X = vectorizer.fit_transform(corpus)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels, kmeans, vectorizer

def analyze_cluster_keywords(cluster_labels, vectorizer, corpus, num_keywords=10):
    """Analyzes top keywords for each cluster."""
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    for cluster in set(cluster_labels):
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        cluster_corpus = [corpus[i] for i in indices]
        cluster_vector = vectorizer.transform(cluster_corpus).toarray().sum(axis=0)
        top_indices = cluster_vector.argsort()[-num_keywords:][::-1]
        cluster_keywords[cluster] = [terms[idx] for idx in top_indices]
    return cluster_keywords

def rank_papers(text_rank_scores, k_truss_scores):
    combined_scores = {node: text_rank_scores[node] + k_truss_scores.get(node, 0) for node in text_rank_scores}
    ranked_papers = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_papers

def generate_pdf(papers, ranked_indices, output_filename="recommended_papers.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Top 25 Recommended Papers", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    for rank, idx in enumerate(ranked_indices[:25], start=1):
        paper = papers[idx]
        pdf.multi_cell(0, 10, f"{rank}. Title: {paper['title']}\n   Authors: {', '.join(paper['authors'])}\n   DOI: {paper['doi']}\n")
        pdf.ln(5)

    pdf.output(output_filename)
    print(f"PDF saved as {output_filename}")

import aiohttp

async def main():
    query = input("Enter a keyword or paper content: ")
    print("Fetching data...")
    
    async with aiohttp.ClientSession() as session:
        try:
            xml_data = await fetch_arxiv_data_with_retries(session, query)
        except Exception as e:
            print(f"Error: {e}")
            return

    if xml_data:
        papers = extract_paper_details(xml_data)
        if not papers:
            print("No papers found.")
        else:
            print("Papers fetched successfully!")

            # Preprocess titles and summaries
            corpus = [preprocess_text(paper["title"] + " " + (paper["summary"] or "")) for paper in papers]

            # Perform clustering
            print("Performing clustering...")
            cluster_labels, kmeans, vectorizer = perform_clustering(corpus, num_clusters=5)
            print(f"Cluster labels: {cluster_labels}")
            print(f"KMeans: {kmeans}")
            print(f"Vectorizer: {vectorizer}")

            # Analyze cluster keywords
            print("Analyzing cluster keywords...")
            cluster_keywords = analyze_cluster_keywords(cluster_labels, vectorizer, corpus)

            # Print cluster keywords
            for cluster, keywords in cluster_keywords.items():
                print(f"Cluster {cluster}: {', '.join(keywords)}")

            # Calculate TextRank scores
            print("Calculating TextRank scores...")
            text_rank_scores, similarity_graph = calculate_text_rank_with_bert(corpus)
            print(f"TextRank scores: {text_rank_scores}")
            print(f"Similarity graph: {similarity_graph}")

            # Calculate k-truss scores
            print("Calculating K-Truss scores...")
            k_truss_graph = calculate_k_truss_features(similarity_graph, k=3)
            k_truss_scores = {node: len(list(k_truss_graph.neighbors(node))) for node in k_truss_graph.nodes}
            print(f"K-Truss scores: {k_truss_scores}")

            # Rank papers
            print("Ranking papers...")
            ranked_papers = rank_papers(text_rank_scores, k_truss_scores)
            print(f"Ranked papers: {ranked_papers}")

            # Generate PDF with ranked papers
            print("Generating PDF...")
            ranked_indices = [idx for idx, _ in ranked_papers]
            generate_pdf(papers, ranked_indices)

            # Plot and save visualizations
            plot_scores(text_rank_scores, "TextRank Scores", "Paper ID", "Score", "text_rank_scores.png")
            plot_scores(k_truss_scores, "K-Truss Scores", "Paper ID", "Score", "k_truss_scores.png")
            plot_clusters(corpus, cluster_labels, "Cluster Visualization", "cluster_visualization.png")

            # Generate detailed report PDF
            generate_report_pdf(papers, [idx for idx, _ in ranked_papers], cluster_labels, text_rank_scores, k_truss_scores)

            # Generate clustered report
            embeddings = vectorizer.transform(corpus).toarray()
            cluster_and_generate_report(papers, embeddings, num_clusters=5)

    else:
        print("Failed to fetch data. Please try again later.")

def plot_scores(scores, title, xlabel, ylabel, filename):
    """
    Plot scores (TextRank or K-Truss) as a bar chart.
    """
    ids, values = zip(*sorted(scores.items(), key=lambda x: x[1], reverse=True))
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(ids)), values, tick_label=ids)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{title} graph saved as {filename}")

def plot_clusters(corpus, cluster_labels, title, filename):
    """
    Visualize clusters with a scatter plot (using 2D embeddings).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(corpus)
    plt.figure(figsize=(10, 6))
    for cluster in set(cluster_labels):
        points = [embeddings[i] for i in range(len(cluster_labels)) if cluster_labels[i] == cluster]
        plt.scatter([p[0] for p in points], [p[1] for p in points], label=f"Cluster {cluster}")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Cluster visualization saved as {filename}")

def create_wordcloud(text, title, filename):
    """
    Generate a word cloud from text.
    """
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Word cloud saved as {filename}")

def generate_report_pdf(papers, ranked_indices, cluster_labels, text_rank_scores, k_truss_scores, output_filename="detailed_report.pdf"):
    """
    Generate a detailed report PDF with visualizations and descriptions.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Detailed Report on Paper Recommendations", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Analysis Overview", ln=True)
    pdf.multi_cell(0, 10, (
        "This report provides an in-depth analysis of the top 25 recommended papers for the given query. "
        "The recommendations are based on TextRank and K-Truss scores, enhanced with clustering and keyword analysis."
    ))
    pdf.ln(5)

    # Include TextRank scores graph
    pdf.add_page()
    pdf.cell(0, 10, "TextRank Scores Visualization", ln=True)
    pdf.image("text_rank_scores.png", x=10, y=None, w=180)
    pdf.ln(100)

    # Include K-Truss scores graph
    pdf.add_page()
    pdf.cell(0, 10, "K-Truss Scores Visualization", ln=True)
    pdf.image("k_truss_scores.png", x=10, y=None, w=180)
    pdf.ln(100)

    # Include Cluster visualization
    pdf.add_page()
    pdf.cell(0, 10, "Cluster Visualization", ln=True)
    pdf.image("cluster_visualization.png", x=10, y=None, w=180)
    pdf.ln(100)

    # Top-ranked papers
    pdf.add_page()
    pdf.cell(0, 10, "Top 25 Recommended Papers", ln=True)
    for rank, idx in enumerate(ranked_indices[:25], start=1):
        paper = papers[idx]
        cluster = cluster_labels[idx]
        pdf.multi_cell(0, 10, (
            f"{rank}. Title: {paper['title']}\n"
            f"   Authors: {', '.join(paper['authors'])}\n"
            f"   DOI: {paper['doi']}\n"
            f"   Cluster: {cluster}\n"
        ))
        pdf.ln(5)

    pdf.output(output_filename)
    print(f"Detailed report saved as {output_filename}")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from fpdf import FPDF

def cluster_and_generate_report(papers, embeddings, num_clusters, output_filename="clustered_report.pdf"):
    """
    Clusters papers using K-Means and adds cluster details to a PDF report with a visualization.

    Args:
        papers (list): List of dictionaries containing paper details (title, authors, DOI, etc.).
        embeddings (ndarray): Embeddings of the papers for clustering.
        num_clusters (int): Number of clusters for K-Means.
        output_filename (str): The name of the output PDF file.

    Returns:
        None
    """
    # Step 1: Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Step 2: Create a scatter plot for cluster visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 7))
    for cluster_id in range(num_clusters):
        points = tsne_results[cluster_labels == cluster_id]
        plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {cluster_id}")
    plt.title("Cluster Visualization")
    plt.legend()
    plt.savefig("cluster_visualization.png")  # Save the visualization for the PDF
    plt.close()

    # Step 3: Group papers by clusters
    clusters = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(papers[idx])

    # Step 4: Generate PDF report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Clustered Papers Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "This report provides clustering results and a visualization of the paper clusters.")
    pdf.ln(10)

    # Add the cluster visualization to the PDF
    pdf.add_page()
    pdf.image("cluster_visualization.png", x=10, y=40, w=190)

    # Add cluster details
    for cluster_id, papers_in_cluster in clusters.items():
        pdf.add_page()
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt=f"Cluster {cluster_id}", ln=True, align="L")
        pdf.ln(5)

        pdf.set_font("Arial", size=12)
        for paper in papers_in_cluster:
            pdf.multi_cell(0, 10, f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nDOI: {paper['doi']}\n")
            pdf.ln(5)

    pdf.output(output_filename)
    print(f"PDF saved as {output_filename}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())