import os
import urllib.request
import tarfile

# Create data directory
os.makedirs("torgo_data", exist_ok=True)

# TORGO download URLs - dysarthric speakers only
# Remove FC and MC (controls) if you only want dysarthric speech
# Include them if you want contrast data
URLS = {
    "F.tar.bz2": "https://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2",   # Females with dysarthria
    "M.tar.bz2": "https://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2",   # Males with dysarthria
    # Uncomment below to also get control speakers (no dysarthria)
    # "FC.tar.bz2": "https://www.cs.toronto.edu/~complingweb/data/TORGO/FC.tar.bz2",
    # "MC.tar.bz2": "https://www.cs.toronto.edu/~complingweb/data/TORGO/MC.tar.bz2",
}

def download_with_progress(url, dest):
    """Download a file showing progress."""
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r  Downloading... {percent}%", end="", flush=True)

    print(f"Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\n  Done: {dest}")

def extract(filepath, dest_dir):
    """Extract a .tar.bz2 file."""
    print(f"Extracting {os.path.basename(filepath)}...")
    with tarfile.open(filepath, "r:bz2") as tar:
        tar.extractall(dest_dir)
    print(f"  Extracted to {dest_dir}/")

# Download and extract each archive
for filename, url in URLS.items():
    local_path = os.path.join("torgo_data", filename)

    if not os.path.exists(local_path):
        download_with_progress(url, local_path)
    else:
        print(f"Already downloaded: {filename}")

    extract(local_path, "torgo_data")

print("\nAll done! Data is in ./torgo_data/")
print("Speakers with dysarthria: F01, F03, F04, M01, M02, M03, M04, M05")
