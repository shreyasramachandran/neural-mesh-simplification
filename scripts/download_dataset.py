import requests
import os
import argparse

# Base URL for Google Drive API
BASE_URL = 'https://www.googleapis.com/drive/v3/files'

def list_files_in_folder(folder_id, api_key):
    """List all files in a Google Drive folder"""
    params = {
        'q': f"'{folder_id}' in parents and trashed = false",
        'key': api_key,
        'fields': 'files(id, name, mimeType)'
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json().get('files', [])
    else:
        print(f"Error fetching files: {response.content}")
        return []

def download_file(file_id, file_name, output_dir, api_key):
    """Download a file from Google Drive by file ID"""
    url = f'{BASE_URL}/{file_id}?alt=media&key={api_key}'
    response = requests.get(url, stream=True)
    file_path = os.path.join(output_dir, file_name)
    
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f'Downloaded {file_name} to {file_path}')
    else:
        print(f'Failed to download {file_name}: {response.content}')

def download_folder(folder_id, output_dir, api_key):
    """Download all files from a Google Drive folder"""
    os.makedirs(output_dir, exist_ok=True)
    files = list_files_in_folder(folder_id, api_key)
    for file in files:
        file_name = file['name']
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            subfolder_path = os.path.join(output_dir, file_name)
            download_folder(file['id'], subfolder_path, api_key)  # Recursive for subfolders
        else:
            download_file(file['id'], file_name, output_dir, api_key)

def extract_folder_id_from_link(folder_link):
    """Extract folder ID from the Google Drive folder link"""
    try:
        folder_id = folder_link.split('/folders/')[1].split('?')[0]
        return folder_id
    except IndexError:
        print("Invalid Google Drive link. Ensure it contains '/folders/' and try again.")
        return None

if __name__ == '__main__':
    # Argument parser setup with named arguments
    parser = argparse.ArgumentParser(description='Download a folder from Google Drive using API key.')
    
    parser.add_argument('--api_key', type=str, required=True, help='Your Google Drive API key')
    parser.add_argument('--folder_link', type=str, required=True, help='The Google Drive folder link')
    parser.add_argument('--output_directory', type=str, required=True, help='The directory where the folder will be saved')

    args = parser.parse_args()

    # Extract folder ID from the provided link
    folder_id = extract_folder_id_from_link(args.folder_link)
    
    if folder_id:
        # Download the folder
        download_folder(folder_id, args.output_directory, args.api_key)