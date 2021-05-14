#https://codingandfun.com/python-scraping-how-to-get-sp-500-companies-from-wikipedia/
import bs4 as bs
import requests
import pandas as pd
 
resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
#prints html:
#print(table)
#finds table elements via html <tr>
#print(table.findAll('tr')[1:])

#loop through table rows
    #create empty lists to put stuff into
tickers = []
names = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text    
    tickers.append(ticker)
    #find company names
    name = row.findAll('td')[1].text
    names.append(name)

#removing '\n' from stuff:
tickers = list(map(lambda s: s.strip(), tickers))
names = list(map(lambda s: s.strip(), names))

#make dataframe:
tickerdf = pd.DataFrame(tickers,columns=['ticker'])
namedf = pd.DataFrame(names, columns=['names_1'])

df = pd.concat([tickerdf, namedf], axis=1)
#send to excel:
#df.to_excel("ticker_list.xlsx")
