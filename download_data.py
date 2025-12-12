import requests
import os

urls = {
    "Fake.csv": "https://raw.githubusercontent.com/jyojay/MONU_Project_4/main/Resources/Fake.csv",
    "True.csv": "https://raw.githubusercontent.com/jyojay/MONU_Project_4/main/Resources/True.csv"
}

def download_file(url, filename):
    print(f"Downloading {filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {filename} successfully.")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    for filename, url in urls.items():
        if os.path.exists(filename):
            print(f"{filename} already exists, skipping.")
        else:
            download_file(url, filename)
