import requests
import nltk
import gensim
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter

from pprint import pprint
from gensim import corpora
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

nltk.download('punkt')
nltk.download('stopwords')


# Set up your API key and Custom Search Engine ID
GOOGLE_API_KEY = "AIzaSyBmVwCC9IprWpIEeuw6We0wVXTLNKqp4h4"
CSE_ID = "45566e593fced4a95"


class SERPAnalyze:
    def __init__(self) -> None:
        pass

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
            results.append(
                {"title": item["title"], "url": item["link"], "snippet": item["snippet"]})

        return results

    def scrape_page(self, url):
        """Fetch and extract text content from a PDF URL.

        Args:
            url (str): URL of the PDF file

        Returns:
            str: Extracted text content from the PDF
        """
        print(f"== Scraping PDF from {url} ==")
        try:
            # First, make a HEAD request to check file size
            head_response = requests.head(url)
            content_length = int(
                head_response.headers.get('content-length', 0))

            # Skip if file is larger than 5MB (5 * 1024 * 1024 bytes)
            if content_length > 5 * 1024 * 1024:
                print(
                    f"Skipped: PDF file too large ({content_length / (1024*1024):.1f}MB)")
                return (False, None)

            # Download the PDF file
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code != 200:
                return f"Failed to download PDF: HTTP {response.status_code}"

            # Import PyPDF2 here to handle PDF processing
            import PyPDF2
            from io import BytesIO

            # Create a PDF file object from the downloaded content
            pdf_file = BytesIO(response.content)

            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            return (True, text.strip())

        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"

    def extract_crime_report_features(self, text):
        """
        Return important features from a crime report text.
        """
        print("== Extracting features ==")
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

    def categorize_report_features(self, papers):
        """
        Categorize the extracted features gotten from each paper into distinct groups.
        """
        print("== Categorizing features ==")
        genai.configure(api_key="AIzaSyAkXHTeQpOr0JarRTwwdG1m4ClwQeYpz24")

        # Flatten the list of features across all papers
        all_features = [feature for paper in papers for feature in paper]

        #  Count occurrences of each feature across all papers
        feature_counter = Counter(all_features)

        #  Send the features to Google Gemini for categorization
        prompt = f"Categorize the following crime-reporting paper features into 10 distinct groups: {', '.join(all_features)} \n Return in the JSON format: category: [feature1, feature2, ...]. Your response should only contain the JSON format "
        response = genai.GenerativeModel(
            "gemini-2.0-flash").generate_content(prompt)

        #  Parse the response
        try:
            # Assuming Gemini returns categories in JSON format
            categories = json.loads(response.text.strip(
                '```').strip().strip("json").strip("JSON"))
        except json.JSONDecodeError:
            categories = response.text  # If not in JSON, just use raw response

        print(categories)
        #  Create a dictionary to count papers that contain features from each category
        category_paper_count = {}

        #  Iterate over each category and its features
        for category, features in categories.items():
            count = 0
            # Iterate over each paper
            for paper in papers:
                # If any feature from the category is found in the paper, count it
                if any(feature in paper for feature in features):
                    count += 1
            category_paper_count[category] = count

        #  Visualize the counts
        category_names = list(category_paper_count.keys())
        category_counts = list(category_paper_count.values())

        return category_names, category_counts

    def visualize_categories(self, category_names, category_counts):
        """
        Create a bar chart to visualize the category occurrences.
        """
        print("== Visualizing categories ==")

        # Create a bar chart to visualize category occurrences
        plt.figure(figsize=(10, 5))
        sns.barplot(x=category_names, y=category_counts, palette="Blues")

        # Customize chart
        plt.xlabel("Categories")
        plt.ylabel("Number of Papers")
        plt.title("Number of Papers Containing Features from Each Category")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Show the plot
        plt.show()


if __name__ == "__main__":
    all_features = []
    program = SERPAnalyze()
    # pprint(program.fetch_google_results("deep learning journal papers"))
    serp_results = program.fetch_google_results(
        "crime-reporting papers filetype:pdf")

    # TODO This should be done concurrently
    for result in serp_results:
        status, text = program.scrape_page(result["url"])
        if not status:
            continue
        features = program.extract_crime_report_features(text)
        all_features.append(features)

    category_names, category_counts = program.categorize_report_features(
        all_features)

    program.visualize_categories(category_names, category_counts)
