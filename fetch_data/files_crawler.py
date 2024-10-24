import requests
from bs4 import BeautifulSoup
import os
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import argparse

# Configure logging to log information and errors to 'crawler.log'
def configure_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO)

# Create a requests session with retry logic to handle transient errors
def configure_session(retry_connect, retry_backoff):
    session = requests.Session()
    retry = Retry(connect=retry_connect, backoff_factor=retry_backoff)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def get_pdf_links(page_number, session):
    """
    Fetches PDF links from a given page of the website.

    Args:
        page_number (int): The page number to fetch links from.
        session (requests.Session): The requests session object.

    Returns:
        list: A list of full URLs pointing to PDF files on the page.
    
    Raises:
        requests.exceptions.HTTPError: If the POST request fails.
    """
    url = f'https://juriscassation.cspj.ma/en/Decisions/RechercheDecisionsRes?page={page_number}'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Accept': '*/*',
        'X-Requested-With': 'XMLHttpRequest',
        'User-Agent': 'Mozilla/5.0 (compatible; YourCrawlerName/1.0)'
    }
    data = {
        'X-Requested-With': 'XMLHttpRequest'
    }

    # Send a POST request to retrieve the PDF links for the page
    response = session.post(url, headers=headers, data=data)
    response.raise_for_status()
    
    # Introduce a delay to avoid overloading the server
    time.sleep(0.5)

    # Parse the response content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract PDF links from anchor tags with class 'dis-dec'
    pdf_links = []
    for a_tag in soup.find_all('a', class_='dis-dec'):
        href = a_tag.get('href')
        if href:
            full_url = f'https://juriscassation.cspj.ma{href}'
            pdf_links.append(full_url)
    return pdf_links

def download_pdf(url, file_id, session, output_dir):
    """
    Downloads a PDF from a given URL and saves it to the output directory.

    Args:
        url (str): The URL of the PDF file to download.
        file_id (int): The unique file identifier for the saved PDF.
        session (requests.Session): The requests session object.
        output_dir (str): The directory to save the downloaded PDFs.

    Raises:
        Exception: If the download fails or the file cannot be saved.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; YourCrawlerName/1.0)'
    }
    try:
        # Send a GET request to download the PDF
        response = session.get(url, headers=headers, stream=True)
        response.raise_for_status()

        # Save the downloaded PDF to the specified directory
        file_path = os.path.join(output_dir, f'{file_id}.pdf')
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logging.info(f'Downloaded {file_path}')
    except Exception as e:
        logging.error(f'Failed to download {url}: {e}')

def download_pdfs_concurrently(pdf_links, start_id, session, output_dir, max_workers):
    """
    Downloads PDF files concurrently from a list of URLs.

    Args:
        pdf_links (list): List of PDF URLs to download.
        start_id (int): Starting file ID for naming the downloaded PDFs.
        session (requests.Session): The requests session object.
        output_dir (str): The directory to save the downloaded PDFs.
        max_workers (int): The maximum number of concurrent download threads.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, pdf_link in enumerate(pdf_links):
            file_id = start_id + idx
            executor.submit(download_pdf, pdf_link, file_id, session, output_dir)

def main(total_pages, log_file, output_dir, max_workers, retry_connect, retry_backoff):
    """
    Main function that orchestrates the crawling and downloading process.

    - It iterates through all pages of the website, fetching PDF links and downloading them.
    - Downloads are performed concurrently to optimize efficiency.
    """
    # Configure logging
    configure_logging(log_file)

    # Ensure the output directory exists for saving the downloaded files
    os.makedirs(output_dir, exist_ok=True)

    # Configure the session with retry logic
    session = configure_session(retry_connect, retry_backoff)

    current_file_id = 1  # Start ID for naming downloaded PDF files

    # Loop through each page and download PDFs
    for page_number in range(1, total_pages + 1):
        print(f'Processing page {page_number} of {total_pages}')
        pdf_links = get_pdf_links(page_number, session)
        download_pdfs_concurrently(pdf_links, current_file_id, session, output_dir, max_workers)
        current_file_id += len(pdf_links)  # Increment file ID for next batch of downloads

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web crawler for downloading PDF files.')

    parser.add_argument('--total_pages', type=int, required=True, help='Total number of pages to crawl.')
    parser.add_argument('--log_file', type=str, default='crawler.log', help='Log file path.')
    parser.add_argument('--output_dir', type=str, default='pdf_files', help='Directory to save downloaded PDFs.')
    parser.add_argument('--max_workers', type=int, default=10, help='Number of concurrent download threads.')
    parser.add_argument('--retry_connect', type=int, default=3, help='Number of retry attempts for connections.')
    parser.add_argument('--retry_backoff', type=float, default=0.5, help='Backoff factor for retries.')

    args = parser.parse_args()

    main(args.total_pages, args.log_file, args.output_dir, args.max_workers, args.retry_connect, args.retry_backoff)
