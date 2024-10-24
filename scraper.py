import requests
from bs4 import BeautifulSoup


class Scraper:
    """A scraper that parses political debates and label text as democratic, republican or neutral"""

    # init links
    def __init__(self) -> None:
        self.originalLink = "https://www.debates.org/voter-education/debate-transcripts/"
        self.rootLink = "https://www.debates.org"
        self.transcriptLinks = []

    # creating soup object
    def requestPage(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup

    # getting all links to scrape
    def getAllTranscriptLinks(self):
        soup = self.requestPage(self.originalLink)
        mainContent = soup.find("div", id="content-sm")
        anchorTags = mainContent.find_all("a")

        for tag in anchorTags:
            link = tag.get("href")
            if "http" not in link:
                link = self.rootLink + link
            self.transcriptLinks.append(link)

    def scrape(self):
        self.getAllTranscriptLinks()
