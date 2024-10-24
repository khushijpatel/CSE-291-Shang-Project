from scraper import Scraper

scraper = Scraper()
scraper.scrape()

for link in scraper.transcriptLinks:
    print(link)
