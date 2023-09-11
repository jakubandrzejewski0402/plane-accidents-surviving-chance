import requests
from bs4 import BeautifulSoup
import csv
import time

with open("aviation_accidents.csv", mode="a", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)

    for year in range(1920, 2024):
        i = 1
        while True:
            time.sleep(5)  # due to database server capabilities...
            url = f"https://www.planecrashinfo.com/{year}/{year}-{i}.htm"
            i += 1
            response = requests.get(url)
            if response.status_code == 404:
                break

            html = response.content
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table", {"cellspacing": "0"})
            rows = table.find_all("tr")[1:]

            data = []
            for index, row in enumerate(rows):
                columns = row.find_all("td")
                column_data = columns[1].text.strip()
                if index == 9 or index == 10:
                    column_data = column_data.split()[0]
                if column_data != "?":
                    data.append(column_data)
                else:
                    data.append("")

            writer.writerow(data)
