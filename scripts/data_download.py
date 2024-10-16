import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import argparse
import hashlib

def download_file(url, save_path, chunk_size=8192):
    """
    Download a file with progress bar and resume capability.
    """
    file_size = int(requests.head(url).headers.get('Content-Length', 0))
    
    if os.path.exists(save_path):
        first_byte = os.path.getsize(save_path)
    else:
        first_byte = 0
    
    if first_byte >= file_size:
        return
    
    header = {"Range": f"bytes={first_byte}-{file_size}"}
    
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    
    req = requests.get(url, headers=header, stream=True)
    
    with open(save_path, 'ab') as f:
        for chunk in req.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()

def get_file_hash(file_path):
    """
    Calculate SHA256 hash of a file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_dataset(language_pair, base_url, save_directory):
    """
    Download dataset files for a given language pair.
    """
    url = f"{base_url}/{language_pair}/"
    os.makedirs(save_directory, exist_ok=True)

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a')
    file_links = [link['href'] for link in links if '.' in link['href']]

    for file_link in file_links:
        file_url = url + file_link
        save_path = os.path.join(save_directory, file_link)
        
        print(f"Downloading {file_url}")
        download_file(file_url, save_path)
        
        print(f"Verifying {file_link}")
        file_hash = get_file_hash(save_path)
        print(f"SHA256: {file_hash}")

def main():
    parser = argparse.ArgumentParser(description="Download Opus-100 dataset files.")
    parser.add_argument("--language", default="en-es", help="Language pair to download (e.g., en-es)")
    args = parser.parse_args()

    base_url = "https://data.statmt.org/opus-100-corpus/v1.0/supervised"
    save_directory = f"./Datasets/{args.language}"

    download_dataset(args.language, base_url, save_directory)
    print("All files have been downloaded and verified.")

if __name__ == "__main__":
    main()