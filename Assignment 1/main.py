import requests
import nltk
import gensim
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
        try:
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

            return text.strip()

        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def extract_crime_report_features(self, text):
        """
        Return important features from a crime report text.
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


if __name__ == "__main__":
    program = SERPAnalyze()
    # pprint(program.fetch_google_results("deep learning journal papers"))
    # pprint(program.fetch_google_results(
    # "crime-reporting papers filetype:pdf"))
    # url = "https://djs.maryland.gov/Documents/MD-DJS-Juvenile-Crime-Data-Brief_20230912.pdf"
    # text = program.scrape_page(url)

    with open("crime_report.txt", "r") as f:
        text = f.read()
        pprint(program.extract_crime_report_features(text))
