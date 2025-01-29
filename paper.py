import os
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import re
import spacy
import asyncio
import aiohttp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from fpdf import FPDF

# Set the number of cores to use
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust this number as needed

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Step 1: Fetch data from arXiv API with asyncio for parallel calls
async def fetch_arxiv_data(session, query, start=0, max_results=10):
    query = urllib.parse.quote_plus(query)  # Properly encode the query
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start={start}&max_results={max_results}'
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

async def fetch_data(query):
    async with aiohttp.ClientSession() as session:
        return await fetch_arxiv_data(session, query)

# Step 2: Extract titles and summaries from XML using ElementTree
def extract_titles_and_summaries(xml_data):
    root = ET.fromstring(xml_data)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    titles = [entry.find('atom:title', namespace).text for entry in root.findall('atom:entry', namespace)]
    summaries = [entry.find('atom:summary', namespace).text for entry in root.findall('atom:entry', namespace)]
    return titles, summaries

# Step 3: Preprocess text (simple example, you can expand this)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    return text.strip()

# Step 4: Vectorize text using TF-IDF
def vectorize_text(corpus):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

# Step 5: Train K-Means clustering model
def train_kmeans(X, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

# Step 6: Visualize clusters using t-SNE
def visualize_clusters(X, labels):
    n_samples = X.shape[0]
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("t-SNE visualization of clusters")
    plt.show()

# Generate PDF
def generate_pdf(clusters, keyword_papers, cluster_keywords, paper_keywords, output_filename="clustered_papers.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title Page
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Clustered Papers and Keyword-Related Papers", ln=True, align="C")
    pdf.ln(10)

    # Add Clustered Papers
    pdf.set_font("Arial", size=12)
    for cluster_id, papers in clusters.items():
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt=f"Cluster {cluster_id}", ln=True, align="L")
        pdf.ln(5)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"Top Keywords: {', '.join(cluster_keywords[cluster_id])}")
        pdf.ln(5)
        for priority, paper in enumerate(papers, start=1):
            pdf.set_font("Arial", size=12)
            # Format paper title in APA style
            formatted_paper = f"{priority}. {paper}."
            pdf.multi_cell(0, 10, f"{formatted_paper}\nKeywords: {', '.join(paper_keywords[paper])}")
            pdf.ln(2)
        pdf.ln(5)

    # Add Keyword-Related Papers
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="Keyword-Related Papers", ln=True, align="L")
    pdf.ln(5)
    for priority, paper in enumerate(keyword_papers, start=1):
        pdf.set_font("Arial", size=12)
        # Format paper title in APA style
        formatted_paper = f"{priority}. {paper}."
        pdf.multi_cell(0, 10, formatted_paper)
        pdf.ln(2)

    # Save PDF
    pdf.output(output_filename)
    print(f"PDF saved as {output_filename}")
    # Extract additional metadata (authors, DOI, published year, and conference/meeting)
    def extract_metadata(xml_data):
        root = ET.fromstring(xml_data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        metadata = []
        for entry in root.findall('atom:entry', namespace):
            title = entry.find('atom:title', namespace).text
            authors = [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)]
            doi = entry.find('atom:id', namespace).text
            published_year = entry.find('atom:published', namespace).text[:4]
            conference = entry.find('atom:source', namespace).text if entry.find('atom:source', namespace) is not None else "N/A"
            metadata.append({
                'title': title,
                'authors': authors,
                'doi': doi,
                'published_year': published_year,
                'conference': conference
            })
        return metadata

    # Generate PDF with additional metadata
    def generate_pdf_with_metadata(clusters, keyword_papers, cluster_keywords, paper_keywords, metadata, output_filename="clustered_papers_with_metadata.pdf"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title Page
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, txt="Clustered Papers and Keyword-Related Papers", ln=True, align="C")
        pdf.ln(10)

        # Add Clustered Papers
        pdf.set_font("Arial", size=12)
        for cluster_id, papers in clusters.items():
            pdf.set_font("Arial", style="B", size=14)
            pdf.cell(200, 10, txt=f"Cluster {cluster_id}", ln=True, align="L")
            pdf.ln(5)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"Top Keywords: {', '.join(cluster_keywords[cluster_id])}")
            pdf.ln(5)
            for priority, paper in enumerate(papers, start=1):
                pdf.set_font("Arial", size=12)
                paper_metadata = next((item for item in metadata if item['title'] == paper), None)
                if paper_metadata:
                    authors = ', '.join(paper_metadata['authors'])
                    formatted_paper = f"{priority}. {authors} ({paper_metadata['published_year']}). {paper_metadata['title']}. {paper_metadata['conference']}. DOI: {paper_metadata['doi']}"
                    pdf.multi_cell(0, 10, f"{formatted_paper}\nKeywords: {', '.join(paper_keywords[paper])}")
                pdf.ln(2)
            pdf.ln(5)

        # Add Keyword-Related Papers
        pdf.add_page()
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt="Keyword-Related Papers", ln=True, align="L")
        pdf.ln(5)
        for priority, paper in enumerate(keyword_papers, start=1):
            pdf.set_font("Arial", size=12)
            paper_metadata = next((item for item in metadata if item['title'] == paper), None)
            if paper_metadata:
                authors = ', '.join(paper_metadata['authors'])
                formatted_paper = f"{priority}. {authors} ({paper_metadata['published_year']}). {paper_metadata['title']}. {paper_metadata['conference']}. DOI: {paper_metadata['doi']} ."
                pdf.multi_cell(0, 10, formatted_paper)
            pdf.ln(2)

        # Save PDF
        pdf.output(output_filename)
        print(f"PDF saved as {output_filename}")

    # Extract metadata
    metadata = extract_metadata(xml_data)

    # Generate PDF with metadata
    generate_pdf_with_metadata(clusters, keyword_papers, cluster_keywords, paper_keywords, metadata)

# Filter duplicate rows in TF-IDF matrix
def filter_duplicate_rows(X, corpus, titles):
    _, unique_indices = np.unique(X.toarray(), axis=0, return_index=True)
    X_unique = X[unique_indices]
    corpus_unique = [corpus[i] for i in unique_indices]
    titles_unique = [titles[i] for i in unique_indices]
    return X_unique, corpus_unique, titles_unique

# Determine optimal number of clusters using silhouette score
def determine_optimal_clusters(X):
    silhouette_scores = []
    n_samples = X.shape[0]
    for n_clusters in range(2, min(11, n_samples)):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
    return optimal_clusters

# Analyze top keywords for each cluster
def analyze_cluster_keywords(cluster_labels, vectorizer, titles, corpus, num_keywords=10):
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    paper_keywords = {}
    for cluster in set(cluster_labels):
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        cluster_corpus = [corpus[i] for i in indices]
        cluster_vector = vectorizer.transform(cluster_corpus).toarray().sum(axis=0)
        top_indices = cluster_vector.argsort()[-num_keywords:][::-1]
        cluster_keywords[cluster] = [terms[idx] for idx in top_indices]
        for i in indices:
            paper_vector = vectorizer.transform([corpus[i]]).toarray().flatten()
            top_paper_indices = paper_vector.argsort()[-num_keywords:][::-1]
            paper_keywords[titles[i]] = [terms[idx] for idx in top_paper_indices]
    return cluster_keywords, paper_keywords

# Main function
if __name__ == "__main__":
    query = input("Enter a keyword or paper content: ")
    print("Fetching data...")
    
    xml_data = asyncio.run(fetch_data(query))
    
    if xml_data:
        print("Extracting titles and summaries...")
        titles, summaries = extract_titles_and_summaries(xml_data)
        print(f"Titles: {titles}")

        for title in titles:
            print(f"Title: {title}")

        # Use only titles for preprocessing
        corpus = [
            preprocess_text(title)
            for title in titles
            if isinstance(title, str) and title.strip()
        ]
        corpus = [doc for doc in corpus if doc]  # Filter out empty documents

        if not corpus:
            print("Corpus is empty. Titles may be missing.")
        else:
            print("Vectorizing text...")
            X, vectorizer = vectorize_text(corpus)

            if X is not None:
                print("Filtering duplicate rows in TF-IDF matrix...")
                X, corpus, titles = filter_duplicate_rows(X, corpus, titles)

                print("Determining optimal number of clusters...")
                optimal_clusters = determine_optimal_clusters(X)
                print(f"Optimal number of clusters: {optimal_clusters}")

                print("Training K-Means clustering model...")
                kmeans = train_kmeans(X, num_clusters=optimal_clusters)

                print("Clustering results:")
                clusters = {i: [] for i in range(optimal_clusters)}
                for idx, label in enumerate(kmeans.labels_):
                    clusters[label].append(titles[idx])
                    print(f"Cluster {label}: {titles[idx]}")

                # Analyze cluster keywords
                cluster_keywords, paper_keywords = analyze_cluster_keywords(kmeans.labels_, vectorizer, titles, corpus)
                print("\nTop keywords per cluster:")
                for cluster, keywords in cluster_keywords.items():
                    print(f"Cluster {cluster}: {', '.join(keywords)}")

                # Visualizations
                visualize_clusters(X.toarray(), kmeans.labels_)

                # Example data
                keyword_papers = titles[:25]  # Example: Top 25 papers related to the keyword

                # Generate PDF
                generate_pdf(clusters, keyword_papers, cluster_keywords, paper_keywords)
            else:
                print("Failed to vectorize text.")
    else:
        print("Failed to fetch data. Please try again later.")