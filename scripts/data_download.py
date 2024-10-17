import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import argparse
import hashlib
import urllib.request

def download_file(url, save_path, chunk_size=8192):
    """
    download a file with progress bar and resume capability.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path where the file will be saved.
        chunk_size (int): The size of each chunk to download (default: 8192 bytes).

    Returns:
        None
    """
    # Get the file size from the Content-Length header
    file_size = int(requests.head(url).headers.get('Content-Length', 0))
    
    # Check if the file already exists and get its current size
    if os.path.exists(save_path):
        first_byte = os.path.getsize(save_path)
    else:
        first_byte = 0
    
    # If the file is already fully downloaded, return
    if first_byte >= file_size:
        return
    
    # Set the range header to resume download from the last byte
    header = {"Range": f"bytes={first_byte}-{file_size}"}
    
    # Initialize the progress bar
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    
    # Start the download
    req = requests.get(url, headers=header, stream=True)
    
    # Write the file in chunks, updating the progress bar
    with open(save_path, 'ab') as f:
        for chunk in req.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()

def get_file_hash(file_path):
    """
    calculate the SHA256 hash of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The SHA256 hash of the file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_dataset(language_pair, base_url, save_directory):
    url = f"{base_url}{language_pair}/"
    print(f"Fetching data from URL: {url}")
    os.makedirs(save_directory, exist_ok=True)
    print(f"Saving files to directory: {save_directory}")

    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a')
    file_links = [link['href'] for link in links if '.' in link['href']]
    
    print(f"Found {len(file_links)} files to download:")
    for link in file_links:
        print(link)

    for file_link in file_links:
        file_url = url + file_link
        save_path = os.path.join(save_directory, file_link)
        
        print(f"Downloading {file_url}")
        print(f"Saving to {save_path}")
        try:
            download_file(file_url, save_path)
        except Exception as e:
            print(f"Error downloading {file_url}: {e}")
            continue
        
        print(f"Verifying {save_path}")
        try:
            file_hash = get_file_hash(save_path)
            print(f"SHA256: {file_hash}")
        except FileNotFoundError:
            print(f"Error: File not found at {save_path}")
        except Exception as e:
            print(f"Error verifying {save_path}: {e}")

def main():
    """
    main function to parse arguments and initiate the dataset download.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download dataset for language pair")
    parser.add_argument("--language", type=str, required=True, help="Language pair (e.g., en-de)")
    args = parser.parse_args()

    # Set the base URL and save directory
    base_url = "https://data.statmt.org/opus-100-corpus/v1.0/supervised/"
    save_directory = os.path.join(os.getcwd(), "../data/raw/", args.language)
    
    print(f"Base URL: {base_url}")
    print(f"Save directory: {save_directory}")
    
    os.makedirs(save_directory, exist_ok=True)
    
    # Start the download process
    download_dataset(args.language, base_url, save_directory)
    print("All files have been downloaded and verified.")

if __name__ == "__main__":
    main()
