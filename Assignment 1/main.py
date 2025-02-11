import re
import requests
import nltk
import PyPDF2
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import json
import concurrent.futures
import logging
from io import BytesIO
from collections import Counter
from pprint import pprint
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

# Disable PyPDF2 unwanted logs
logger = logging.getLogger("PyPDF2")
logger.setLevel(logging.ERROR)

nltk.download('punkt')
nltk.download('stopwords')


# Set up Google API key and Custom Search Engine ID
GOOGLE_API_KEY = "AIzaSyBmVwCC9IprWpIEeuw6We0wVXTLNKqp4h4"
CSE_ID = "45566e593fced4a95"


class SERPAnalyze:
    def fetch_google_results(self, query):
        """Fetch top search results using Google Custom Search API."""

        print(f"== Fetching results for {query} ==")
        url = f"https://www.googleapis.com/customsearch/v1"
        params = {"q": query, "key": GOOGLE_API_KEY, "cx": CSE_ID, "num": 10}

        response = requests.get(url, params=params)
        data = response.json()

        # Extract useful information
        results = []
        for item in data.get("items", []):
            results.append(item['link'])

        return results

    def scrape_page(self, url):
        """Fetch and extract text content from a PDF URL."""
        print(f"== Scraping PDF from {url} ==")
        try:
            # First, make a HEAD request to check file size
            head_response = requests.head(url)
            content_length = int(
                head_response.headers.get('content-length', 0))

            # Skip if file is larger than 5MB
            if content_length > 5 * 1024 * 1024:
                # print(
                #     f"Skipped: PDF file too large ({content_length / (1024*1024):.1f}MB)")
                return (False, None)

            response = requests.get(url)
            if response.status_code != 200:
                return False, f"Failed to download PDF: HTTP {response.status_code}"

            pdf_file = BytesIO(response.content)

            # Add error handling for PDF extraction
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    try:
                        text += page.extract_text() + "\n"
                    except Exception:
                        continue  # Skip problematic pages

                if not text.strip():
                    return False, "No text could be extracted from PDF"

                return (True, text.strip())
            except Exception as e:
                return False, f"Error reading PDF: {str(e)}"

        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"

    def fetch_all_features(self, serp_results):
        """ Concurrently fetch and extract text content from all PDF URLs in the search results."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return list(executor.map(self.scrape_page, serp_results))

    def extract_research_paper_features(self, text):
        """
        Return important features from the text content of a research paper.
        """
        # Tokenize and clean the text
        sentences = sent_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))

        # Process for bigrams
        all_tokens = []
        clean_tokens = []
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            all_tokens.extend(tokens)
            # Keep tokens that are alphanumeric and not stopwords
            clean_tokens.extend([word for word in tokens
                                if word.isalnum() and word not in stop_words])

        # Find bigram collocations
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(all_tokens)
        # Filter out bigrams with stopwords or punctuation
        finder.apply_word_filter(lambda w: w in stop_words or not w.isalnum())
        # Get top bigrams using likelihood ratio
        bigrams = finder.nbest(bigram_measures.likelihood_ratio, 10)

        # Get important single words
        word_freq = {}
        for token in clean_tokens:
            word_freq[token] = word_freq.get(token, 0) + 1

        # Combine features
        features = []
        # Add bigram phrases
        features.extend([' '.join(bigram) for bigram in bigrams])
        # Add important single words
        single_words = sorted(
            word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        features.extend([word for word, _ in single_words
                        if not any(word in bigram for bigram in features)])

        return features

    def categorize_features(self, features, type="crime-report"):
        """
        Use Gemini API to categorize the extracted features gotten from each paper into distinct groups.
        """
        print("== Categorizing features ==")
        genai.configure(api_key="AIzaSyAkXHTeQpOr0JarRTwwdG1m4ClwQeYpz24")

        # Flatten the list of features across all papers
        all_features = [feature for paper in features for feature in paper]

        # Create a prompt for Gemini based on the type of papers
        prompt = f"Categorize the following {'crime-reporting paper features' if type == 'crime-report' else 'deep learning models sub-headings'} into 10 distinct groups: {', '.join(all_features)} \n Return in the JSON format: category: [feature1, feature2, ...]. Your response should only contain the JSON format "

        #  Send the features to Google Gemini for categorization
        response = genai.GenerativeModel(
            "gemini-2.0-flash").generate_content(prompt)

        #  Parse the response
        try:
            # Assuming Gemini returns categories in JSON format
            categories = json.loads(response.text.strip(
                '```').strip().strip("json").strip("JSON"))
        except json.JSONDecodeError:
            categories = response.text  # If not in JSON, just use raw response

        #  Create a dictionary to count papers that contain features from each category
        category_paper_count = {}

        #  Iterate over each category and its features
        for category, features in categories.items():
            count = 0
            # Iterate over each paper
            for paper in features:
                # If any feature from the category is found in the paper, count it
                if any(feature in paper for feature in features):
                    count += 1
            category_paper_count[category] = count

        category_names = list(category_paper_count.keys())
        category_counts = list(category_paper_count.values())

        return category_names, category_counts

    def visualize_crime_report(self, category_names, category_counts):
        """
        Create a bar chart to visualize the category occurrences.
        """
        print("== Visualizing Crime Reports ==")

        # Create a bar chart to visualize category occurrences
        plt.figure(figsize=(10, 5))
        sns.barplot(x=category_names, y=category_counts)

        # Customize chart
        plt.xlabel("Categories")
        plt.ylabel("Number of Papers")
        plt.title("Number of Papers Containing Features from Each Category")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Show the plot
        plt.show()

    def visualize_dl_headings(self, category_names, category_counts):
        """
        Create a bar chart to visualize the occurrences of sub-headings in Deep learning models papers.
        """
        print("== Visualizing Deep Learning Sub-Headings ==")

        plt.figure(figsize=(10, 5))
        sns.barplot(x=category_names, y=category_counts)

        # Customize chart
        plt.xlabel("Deep Learning Sub-Headings")
        plt.ylabel("Number of Papers")
        plt.title("Deep Learning Related Sub-Headings in Journal Papers")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.show()

    def analyse(self, paper="crime-report"):
        if paper == "crime-report":
            query = "crime-reporting papers filetype:pdf"
        else:
            query = "deep learning models journal papers filetype:pdf"

        serp_result_urls = self.fetch_google_results(query)

        print("== Fetching and extracting text content from PDFs ==")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Concurrently fetch and extract text content from all PDF URLs in the search results
            paper_contents = list(executor.map(
                self.scrape_page, serp_result_urls))
            # Filter only successful results where status is True
            paper_contents = [content for status,
                              content in paper_contents if status]

        print("== Extracting all research paper features ==")
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            all_features = list(executor.map(
                self.extract_research_paper_features, paper_contents))

        category_names, category_counts = self.categorize_features(
            all_features, type=paper)

        if paper == "crime-report":
            self.visualize_crime_report(category_names, category_counts)
        else:
            self.visualize_dl_headings(category_names, category_counts)

    def run(self):
        print("== Running SERP Analyze ==")

        print("##### Crime Report Analysis #####")
        self.analyse("crime-report")

        print("##### Deep Learning Models Analysis #####")
        self.analyse("deep-learning")


if __name__ == "__main__":
    program = SERPAnalyze()
    program.run()
