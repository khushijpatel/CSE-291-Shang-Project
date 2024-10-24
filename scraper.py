import requests
from bs4 import BeautifulSoup
from data import candidateData
import pandas as pd


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
        self.data = []
        self.labels = []
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

    def writeToFile(self, file, speaker, line):
        with open(file, 'a') as f:
            f.write(speaker + ": " + line + "\n\n")

    def addData(self, party, text):
        self.labels.append(party)
        self.data.append(text)

    # gets the political party of speaker
    def getSpeakerParty(self, speaker):

        lastName = speaker.strip().split()[-1].lower()
        if lastName in candidateData:
            return candidateData[lastName]
        return "Neutral"

    def splitText(self, line, delim):
        text = line.text.split(delim)

        # handling occurances when there is colon in main response
        if len(text) > 1 and len(text[0]) > 20:
            text = [delim.join(text)]
        return text

    def parseText(self, soup):
        transcript = soup.find_all("p")

        currentSpeaker = ""
        speakerResponse = ""

        for line in transcript:
            text = self.splitText(line, ':')

            # checks if new speaker
            if len(text) > 1:

                # only adds data if currentSpeaker adn their response exists
                if currentSpeaker and speakerResponse:
                    # self.writeToFile("test.txt", currentSpeaker, speakerResponse)
                    self.addData(party=currentSpeaker, text=speakerResponse)

                speaker, response = text[0], text[1]
                currentSpeaker = self.getSpeakerParty(speaker)
                speakerResponse = response.strip()
            else:
                speakerResponse += " " + text[0]

        # self.writeToFile("test.txt", currentSpeaker, speakerResponse)

        # adding last speaker and text
        self.addData(party=currentSpeaker, text=speakerResponse)

    def parseLinks(self):
        for i in range(0, 3):
            link = self.transcriptLinks[i]

            # link = self.transcriptLinks[0]
            soup = self.requestPage(link)
            self.parseText(soup)

    def scrape(self):
        self.getAllTranscriptLinks()
        self.parseLinks()

    def createCSV(self, fileName):
        df = pd.DataFrame({
            'Text': self.data,
            'Label': self.labels
        })

        # Save to a CSV file
        df.to_csv(fileName, index=False)
