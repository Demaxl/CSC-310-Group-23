import requests
from pprint import pprint


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


if __name__ == "__main__":
    program = SERPAnalyze()
    pprint(program.fetch_google_results("deep learning journal papers"))
