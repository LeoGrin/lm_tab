Copy
import csv 
from pathlib import Path

import pyppeteer

async def main():
    browser = await pyppeteer.launch()
    page = await browser.newPage()

    await page.goto('https://example.com')

    await page.waitForSelector('table') 

    rows = await page.querySelectorAll('table tr')

    data = []
    for row in rows:
        cells = await row.querySelectorAll('td')
        row_data = []
        for cell in cells:
            text = await cell.evaluate(el => el.textContent)
            row_data.append(text)
        
        data.append(row_data)

    with open('table.csv', 'w', newline='') as f:
        writer = csv.writer(f) 
        writer.writerows(data)

    await browser.close()

if __name__ == '__main__':
    import asyncio 
    asyncio.run(main())