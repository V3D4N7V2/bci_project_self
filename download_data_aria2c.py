import sys
import os
import urllib.request
import json
import zipfile
import subprocess


########################################################################################
#
# Helpers.
#
########################################################################################


def aria2_download(url, output_path):
    """
    Download a file using aria2c with 8 connections.
    The output directory must already exist.
    """

    output_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)

    cmd = [
        "aria2c",
        "-x8",                 # 8 connections
        "-s8",                 # 8 segments
        "-k1M",                # 1MB per segment
        "--continue=true",     # resume if partial
        "--max-connection-per-server=8",
        "--dir", output_dir,   # where file goes
        "--out", filename,     # name of output file
        url
    ]

    print(f"\n[aria2c] Downloading: {filename}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


########################################################################################
#
# Main function.
#
########################################################################################


def main():
    DRYAD_DOI = "10.5061/dryad.dncjsxm85"

    DATA_DIR = "data/"
    data_dirpath = os.path.abspath(DATA_DIR)

    assert os.getcwd().endswith(
        "bci_class_project"
    ), f"Please run the download command from the bci_class_project directory (instead of {os.getcwd()})"

    assert os.path.exists(
        data_dirpath
    ), "Cannot find the data directory to download into."

    DRYAD_ROOT = "https://datadryad.org"
    urlified_doi = DRYAD_DOI.replace("/", "%2F")

    # Lookup dataset versions
    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"
    with urllib.request.urlopen(versions_url) as response:
        versions_info = json.loads(response.read().decode())

    # Grab file listings for the latest version
    files_url_path = versions_info["_embedded"]["stash:versions"][-1]["_links"]["stash:files"]["href"]
    files_url = f"{DRYAD_ROOT}{files_url_path}"
    with urllib.request.urlopen(files_url) as response:
        files_info = json.loads(response.read().decode())

    file_infos = files_info["_embedded"]["stash:files"]

    # Download each file and unzip if needed
    for file_info in file_infos:
        filename = file_info["path"]

        if filename == "README.md":
            continue

        download_path = file_info["_links"]["stash:download"]["href"]
        download_url = f"{DRYAD_ROOT}{download_path}"

        download_to_filepath = os.path.join(data_dirpath, filename)

        # Make sure the parent directory exists if there are subfolders
        os.makedirs(os.path.dirname(download_to_filepath), exist_ok=True)

        # Use aria2c instead of urllib
        aria2_download(download_url, download_to_filepath)

        # If the file is a ZIP, extract it
        if file_info.get("mimeType") == "application/zip":
            print(f"Extracting files from {filename} ...")
            with zipfile.ZipFile(download_to_filepath, "r") as zf:
                zf.extractall(data_dirpath)

    print(f"\nDownload complete. See data files in {data_dirpath}\n")


if __name__ == "__main__":
    main()
