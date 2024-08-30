import pdfkit
import os

# Define the path to wkhtmltopdf
path_wkhtmltopdf = r"C:\Users\MLASSOUED\Documents\llam2\wkhtmltox\bin\wkhtmltopdf.exe"

# Add the path for the configuration
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# List of URLs to convert to PDF
urls = ["https://wiki.oddo.fr/pages/viewpage.action?pageId=1140490280"]

# Ensure the 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')



for url in urls:
    # Create a filename from the URL
    filename = url.replace("https://", "").replace("/", "_").replace("?", "_") + ".pdf"

    # Convert and save the content of the URL in the file
    pdfkit.from_url(url, os.path.join('data', filename), configuration=config)




