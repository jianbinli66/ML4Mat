#!/usr/bin/env python3
# pdf_processor.py
import configparser
import os
import json
import time
import zipfile
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import re

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv("MINERU_API_KEY")

# ========= User only needs to modify here =========
POLL_INTERVAL = 10  # Polling interval (seconds)
MAX_FILES_PER_BATCH = 5  # Maximum number of files to process at once
# ==================================================

BASE_URL = "https://mineru.net/api/v4"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}


# ---------- Helper Functions ----------
def find_appendix_file(main_pdf_path: Path) -> Path:
    """Find appendix file for a given main PDF file"""
    # Try different appendix naming patterns
    stem = main_pdf_path.stem
    patterns = [
        f"{stem}_appendix.pdf",
        f"{stem}_supplement.pdf",
        f"{stem}_supplementary.pdf",
        f"{stem}_appendix_material.pdf",
        f"{stem}_supplementary_material.pdf",
    ]

    for pattern in patterns:
        appendix_path = main_pdf_path.parent / pattern
        if appendix_path.exists():
            return appendix_path

    # Also try looking for any file containing the stem and "appendix" or "supplement"
    parent_dir = main_pdf_path.parent
    for file in parent_dir.glob("*"):
        if file.suffix.lower() == ".pdf" and file != main_pdf_path:
            filename_lower = file.name.lower()
            stem_lower = stem.lower()
            if (stem_lower in filename_lower and
                    any(keyword in filename_lower for keyword in ["appendix", "supplement", "supplementary"])):
                return file

    return None


def merge_appendix_to_main(main_dir: Path, appendix_dir: Path) -> bool:
    """Merge appendix content into main article directory, return success status"""
    main_full_md = main_dir / "full.md"
    appendix_full_md = appendix_dir / "full.md"

    if not main_full_md.exists():
        print(f"⚠️ Main full.md not found in {main_dir}")
        return False

    if not appendix_full_md.exists():
        print(f"⚠️ Appendix full.md not found in {appendix_dir}")
        return False

    try:
        # Read main content
        with open(main_full_md, 'r', encoding='utf-8') as f:
            main_content = f.read()

        # Read appendix content
        with open(appendix_full_md, 'r', encoding='utf-8') as f:
            appendix_content = f.read()

        # Add appendix section to main content
        merged_content = main_content + "\n\n# Appendix\n\n" + appendix_content

        # Write merged content back
        with open(main_full_md, 'w', encoding='utf-8') as f:
            f.write(merged_content)

        # Copy images from appendix to main directory
        appendix_images_dir = appendix_dir / "images"
        main_images_dir = main_dir / "images"

        if appendix_images_dir.exists() and appendix_images_dir.is_dir():
            # Ensure main images directory exists
            main_images_dir.mkdir(exist_ok=True)

            # Copy all image files
            for image_file in appendix_images_dir.glob("*"):
                if image_file.is_file():
                    dest_file = main_images_dir / image_file.name
                    # Handle duplicate filenames
                    counter = 1
                    while dest_file.exists():
                        name_parts = image_file.stem.rsplit('_', 1)
                        if len(name_parts) > 1 and name_parts[1].isdigit():
                            base_name = name_parts[0]
                        else:
                            base_name = image_file.stem
                        new_name = f"{base_name}_appendix_{counter}{image_file.suffix}"
                        dest_file = main_images_dir / new_name
                        counter += 1

                    # Copy the file
                    with open(image_file, 'rb') as src, open(dest_file, 'wb') as dst:
                        dst.write(src.read())

        return True

    except Exception as e:
        print(f"❌ Error merging appendix content: {e}")
        return False


def delete_appendix_dir(appendix_dir: Path) -> bool:
    """Delete appendix directory after successful merge"""
    try:
        if appendix_dir.exists() and appendix_dir.is_dir():
            shutil.rmtree(appendix_dir)
            print(f"🗑️  Deleted appendix directory: {appendix_dir}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error deleting appendix directory {appendix_dir}: {e}")
        return False


# ---------- 1. Upload ----------
def upload_batch(pdf_files: List[Path]) -> str:
    """Upload specified PDF files, return batch_id"""
    if not pdf_files:
        raise RuntimeError("No PDF files provided for upload")

    # Ensure we only process MAX_FILES_PER_BATCH files
    if len(pdf_files) > MAX_FILES_PER_BATCH:
        pdf_files = pdf_files[:MAX_FILES_PER_BATCH]

    print(f"📤 Preparing to upload {len(pdf_files)} PDF file(s)")

    files_data = [
        {"name": p.name, "is_ocr": True, "data_id": f"{p.stem}.pdf-id"}
        for p in pdf_files
    ]
    payload = {
        # "enable_formula": False,
        "language": "en",
        # "enable_table": True,
        "is_ocr":True,
        "model_version":"vlm",
        "files": files_data
    }

    url = f"{BASE_URL}/file-urls/batch"
    r = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    resp = r.json()
    if resp.get("code") != 0:
        raise RuntimeError(f"Failed to get upload URLs: {resp}")

    data = resp["data"]
    batch_id = data["batch_id"]
    upload_urls: List[str] = data["file_urls"]

    print(f"📦 Batch ID: {batch_id}")

    # Upload files one by one using PUT
    for pdf_path, upload_url in zip(pdf_files, upload_urls):
        with open(pdf_path, "rb") as f:
            up_resp = requests.put(upload_url, data=f)
            if up_resp.status_code != 200:
                raise RuntimeError(f"Failed to upload {pdf_path.name} - Status code: {up_resp.status_code}")
        print(f"✅ Successfully uploaded {pdf_path.name}")

    return batch_id


# ---------- 2. Polling ----------
def wait_until_done(batch_id: str) -> List[Dict]:
    """Poll until all tasks are completed, return extract_result list"""
    url = f"{BASE_URL}/extract-results/batch/{batch_id}"
    while True:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") != 0:
            raise RuntimeError(f"Failed to query results: {resp}")

        results: List[Dict] = resp["data"]["extract_result"]
        states = {item["state"] for item in results}
        print(f"🔄 Current states: {states}")

        # Check if all tasks are in final states
        if states <= {"done", "failed", "error"}:
            return results
        time.sleep(POLL_INTERVAL)


# ---------- 3. Batch Download ----------
def download_zip(url: str, dst: Path) -> None:
    """Download and extract zip file"""
    print(f"⬇️  Downloading {url}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    zip_path = dst.with_suffix(".zip")
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract zip file
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dst)
    zip_path.unlink()  # Delete zip file
    print(f"✅ Extracted to {dst}")


def fetch_and_download(batch_id: str, results: List[Dict], output_dir: Path, is_appendix: bool = False) -> None:
    """Download successful zip files"""
    batch_out = output_dir
    batch_out.mkdir(exist_ok=True)

    for item in results:
        if item["state"] != "done":
            print(f"⚠️ {item['file_name']} processing failed: {item['err_msg']}")
            continue
        zip_url = item["full_zip_url"].strip()
        data_id = item["data_id"]

        # Add suffix for appendix files
        if is_appendix:
            data_id = f"{data_id}_appendix"

        target_dir = batch_out / data_id
        download_zip(zip_url, target_dir)


def delete_sections_from_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Build a comprehensive regex pattern that matches all sections at any heading level
    section_keywords = [
        'References',
        'CRediT\\s*(authorship|contribution)\\s*statement',
        'Declaration\\s*of\\s*(competing\\s*interest|competing\\s*financial\\s*interest)',
        'Data\\s*availability',
        'Acknowledgments',
        'Acknowledgements',
    ]

    # Create pattern: match any heading level (1-3 #) followed by any section keyword
    # Capture everything until next heading or end of file
    pattern = r'(?:^|\n)#{1,3}\s*(?:' + '|'.join(section_keywords) + r')\b.*?(?=(?:\n#{1,3}\s|\Z))'

    new_content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)

    if new_content != content:
        # Clean up excessive blank lines
        new_content = re.sub(r'\n{3,}', '\n\n', new_content.strip())

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ Removed specified sections from: {file_path}")
        return True
    else:
        print(f"ℹ️  No sections to remove found in: {file_path}")
        return False

def process_of_section_delete(root_dir):
    processed = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md') and file == 'full.md':
                file_path = os.path.join(root, file)
                print(f"\n📄 Processing: {file_path}")
                if delete_sections_from_md(file_path):
                    processed += 1

    print(f"\n✅ Done! SECTION DELETE Processed {processed} files.")
# ---------- Main Process ----------
def process_files_in_batches(input_dir: Path, output_dir: Path, papers: list = None):
    """Process PDF files and their appendices in batches of MAX_FILES_PER_BATCH"""
    if not TOKEN or TOKEN == "API token applied from website":
        raise RuntimeError("Please set TOKEN first")

    # Get all PDF files
    all_pdf_files = list(input_dir.glob("*.pdf"))
    if not all_pdf_files:
        raise RuntimeError("No PDF files found in input directory")

    # Limit papers if specified
    if papers:
        all_pdf_files = []
        for paper_id in papers:
            file_path = Path(os.path.join(input_dir, str(paper_id) + '.pdf'))
            all_pdf_files.append(file_path)

    print(f"📄 Found {len(all_pdf_files)} PDF file(s) in total")

    # Remove PDFs that are already extracted
    all_pdf_files = [f for f in all_pdf_files if not (output_dir / f"{f.stem}.pdf-id").exists()]
    print(f"📄 {len(all_pdf_files)} PDF file(s) to be processed after filtering extracted ones")

    # Find appendix files
    main_to_appendix = {}
    appendix_files = []

    for pdf_file in all_pdf_files.copy():  # Use copy to avoid modification during iteration
        appendix_file = find_appendix_file(pdf_file)
        if appendix_file and appendix_file not in appendix_files:
            main_to_appendix[pdf_file] = appendix_file
            appendix_files.append(appendix_file)
            # Remove appendix from main file list to avoid double processing
            if appendix_file in all_pdf_files:
                all_pdf_files.remove(appendix_file)

    print(f"🔍 Found {len(appendix_files)} appendix file(s)")

    # Process main files in batches
    for i in range(0, len(all_pdf_files), MAX_FILES_PER_BATCH):
        batch_files = all_pdf_files[i:i + MAX_FILES_PER_BATCH]
        batch_number = (i // MAX_FILES_PER_BATCH) + 1
        total_batches = (len(all_pdf_files) + MAX_FILES_PER_BATCH - 1) // MAX_FILES_PER_BATCH

        print(f"\n{'=' * 50}")
        print(f"Processing Main Files - Batch {batch_number}/{total_batches}")
        print(f"Files: {[f.name for f in batch_files]}")
        print(f"{'=' * 50}")

        try:
            print("=== Starting Upload ===")
            batch_id = upload_batch(batch_files)

            print("=== Waiting for Processing to Complete ===")
            results = wait_until_done(batch_id)

            print("=== Starting to Download Results ===")
            fetch_and_download(batch_id, results, output_dir)

            print(f"✅ Main Batch {batch_number}/{total_batches} completed successfully")

        except Exception as e:
            print(f"❌ Main Batch {batch_number}/{total_batches} failed: {e}")
            # Continue with next batch even if current batch fails
            continue

        # Optional: Add delay between batches to avoid rate limiting
        if i + MAX_FILES_PER_BATCH < len(all_pdf_files):
            print(f"⏳ Waiting before processing next batch...")
            time.sleep(5)
    for file in all_pdf_files:

        dir = os.path.join(output_dir,  f"{file.stem}.pdf-id")
        print(dir)
        # delete references
        process_of_section_delete(dir)
    # Process appendix files in batches
    if appendix_files:
        print(f"\n{'=' * 50}")
        print(f"Processing {len(appendix_files)} Appendix Files")
        print(f"{'=' * 50}")

        for i in range(0, len(appendix_files), MAX_FILES_PER_BATCH):
            batch_files = appendix_files[i:i + MAX_FILES_PER_BATCH]
            batch_number = (i // MAX_FILES_PER_BATCH) + 1
            total_batches = (len(appendix_files) + MAX_FILES_PER_BATCH - 1) // MAX_FILES_PER_BATCH

            print(f"\n{'=' * 50}")
            print(f"Processing Appendix Files - Batch {batch_number}/{total_batches}")
            print(f"Files: {[f.name for f in batch_files]}")
            print(f"{'=' * 50}")

            try:
                print("=== Starting Upload ===")
                batch_id = upload_batch(batch_files)

                print("=== Waiting for Processing to Complete ===")
                results = wait_until_done(batch_id)

                print("=== Starting to Download Results ===")
                fetch_and_download(batch_id, results, output_dir, is_appendix=True)

                print(f"✅ Appendix Batch {batch_number}/{total_batches} completed successfully")

            except Exception as e:
                print(f"❌ Appendix Batch {batch_number}/{total_batches} failed: {e}")
                # Continue with next batch even if current batch fails
                continue

            # Optional: Add delay between batches to avoid rate limiting
            if i + MAX_FILES_PER_BATCH < len(appendix_files):
                print(f"⏳ Waiting before processing next batch...")
                time.sleep(5)

    # Merge appendix content into main articles and delete appendix directories
    print(f"\n{'=' * 50}")
    print("Merging Appendix Content into Main Articles")
    print(f"{'=' * 50}")

    merged_count = 0
    deleted_count = 0
    for main_pdf, appendix_pdf in main_to_appendix.items():
        main_dir = output_dir / f"{main_pdf.stem}.pdf-id"
        appendix_dir = output_dir / f"{appendix_pdf.stem}.pdf-id_appendix"

        if main_dir.exists() and appendix_dir.exists():
            try:
                # Merge appendix content into main
                if merge_appendix_to_main(main_dir, appendix_dir):
                    print(f"✅ Merged appendix for {main_pdf.name}")
                    merged_count += 1

                    # Delete appendix directory after successful merge
                    if delete_appendix_dir(appendix_dir):
                        deleted_count += 1
                else:
                    print(f"⚠️ Failed to merge appendix for {main_pdf.name}")
            except Exception as e:
                print(f"❌ Error processing appendix for {main_pdf.name}: {e}")
        else:
            print(
                f"⚠️ Missing directories for {main_pdf.name}: main={main_dir.exists()}, appendix={appendix_dir.exists()}")

    print(f"\n🎉 All processing completed!")
    print(f"   - Processed {len(all_pdf_files)} main file(s)")
    print(f"   - Processed {len(appendix_files)} appendix file(s)")
    print(f"   - Merged {merged_count} appendix files into their main articles")
    print(f"   - Deleted {deleted_count} appendix directories")