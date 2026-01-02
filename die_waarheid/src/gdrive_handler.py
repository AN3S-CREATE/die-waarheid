"""
Google Drive Integration for Die Waarheid
Handles OAuth authentication and file operations
"""

import os
import io
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

from config import (
    GDRIVE_SCOPES,
    GOOGLE_CREDENTIALS_PATH,
    TOKEN_PICKLE_PATH,
    GDRIVE_AUDIO_FOLDER,
    GDRIVE_TEXT_FOLDER,
    AUDIO_DIR,
    TEXT_DIR,
    SUPPORTED_AUDIO_FORMATS
)

logger = logging.getLogger(__name__)


class GDriveHandler:
    """
    Manages Google Drive authentication and file operations
    Implements OAuth 2.0 flow with token persistence
    """

    def __init__(self):
        self.credentials_path = GOOGLE_CREDENTIALS_PATH
        self.token_path = TOKEN_PICKLE_PATH
        self.service = None
        self.authenticated = False
        self.creds = None

    def authenticate(self) -> Tuple[bool, str]:
        """
        Authenticate with Google Drive API using OAuth 2.0

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if self.token_path.exists():
                with open(self.token_path, 'rb') as token:
                    self.creds = pickle.load(token)
                logger.info("Loaded existing credentials from token.pickle")

            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                    logger.info("Refreshed expired credentials")
                else:
                    if not self.credentials_path.exists():
                        return False, (
                            "❌ Google credentials file not found!\n\n"
                            "**Setup Instructions:**\n"
                            "1. Go to https://console.cloud.google.com/\n"
                            "2. Create a project and enable Google Drive API\n"
                            "3. Create OAuth 2.0 credentials (Desktop app)\n"
                            "4. Download JSON and save as `credentials/google_credentials.json`"
                        )

                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            str(self.credentials_path),
                            GDRIVE_SCOPES
                        )
                        self.creds = flow.run_local_server(port=0)
                        logger.info("Completed OAuth 2.0 authentication flow")
                    except Exception as e:
                        logger.error(f"Authentication flow failed: {str(e)}")
                        return False, f"❌ Authentication failed: {str(e)}"

                self.token_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.token_path, 'wb') as token:
                    pickle.dump(self.creds, token)
                logger.info("Saved credentials to token.pickle")

            try:
                self.service = build('drive', 'v3', credentials=self.creds)
                self.authenticated = True
                logger.info("Successfully built Google Drive service")
                return True, "✅ Successfully authenticated with Google Drive"
            except Exception as e:
                logger.error(f"Failed to build service: {str(e)}")
                return False, f"❌ Failed to build service: {str(e)}"

        except Exception as e:
            logger.error(f"Unexpected error during authentication: {str(e)}")
            return False, f"❌ Unexpected error during authentication: {str(e)}"

    def get_folder_id(self, folder_name: str) -> Optional[str]:
        """
        Get folder ID from folder name

        Args:
            folder_name: Name of the folder to find

        Returns:
            Folder ID if found, None otherwise
        """
        if not self.authenticated:
            logger.warning("Not authenticated - cannot get folder ID")
            return None

        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)',
                pageSize=10
            ).execute()

            items = results.get('files', [])
            if items:
                logger.info(f"Found folder '{folder_name}' with ID: {items[0]['id']}")
                return items[0]['id']
            
            logger.warning(f"Folder '{folder_name}' not found in Google Drive")
            return None

        except HttpError as error:
            logger.error(f"Error finding folder '{folder_name}': {error}")
            return None

    def list_files_in_folder(
        self,
        folder_id: str,
        file_extension: Optional[str] = None
    ) -> List[Dict]:
        """
        List all files in a folder, optionally filtered by extension

        Args:
            folder_id: Google Drive folder ID
            file_extension: Optional file extension filter (e.g., '.opus')

        Returns:
            List of file metadata dictionaries
        """
        if not self.authenticated:
            logger.warning("Not authenticated - cannot list files")
            return []

        try:
            query = f"'{folder_id}' in parents and trashed=false"
            if file_extension:
                query += f" and name contains '{file_extension}'"

            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType, createdTime, modifiedTime, size)',
                pageSize=1000
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in folder {folder_id}")
            return files

        except HttpError as error:
            logger.error(f"Error listing files: {error}")
            return []

    def list_subfolders(self, folder_id: str) -> List[Dict]:
        """
        List all subfolders within a folder

        Args:
            folder_id: Parent folder ID

        Returns:
            List of subfolder metadata dictionaries
        """
        if not self.authenticated:
            logger.warning("Not authenticated - cannot list subfolders")
            return []

        try:
            query = (
                f"'{folder_id}' in parents and "
                "mimeType='application/vnd.google-apps.folder' and "
                "trashed=false"
            )

            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, createdTime)',
                pageSize=1000
            ).execute()

            subfolders = results.get('files', [])
            logger.info(f"Found {len(subfolders)} subfolders in folder {folder_id}")
            return subfolders

        except HttpError as error:
            logger.error(f"Error listing subfolders: {error}")
            return []

    def download_file(
        self,
        file_id: str,
        destination_path: Path,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, str]:
        """
        Download a file from Google Drive

        Args:
            file_id: Google Drive file ID
            destination_path: Local path to save file
            progress_callback: Optional callback function for progress updates

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.authenticated:
            logger.warning("Not authenticated - cannot download file")
            return False, "Not authenticated with Google Drive"

        try:
            request = self.service.files().get_media(fileId=file_id)
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            with io.FileIO(str(destination_path), 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False

                while not done:
                    status, done = downloader.next_chunk()
                    if progress_callback and status:
                        progress_callback(int(status.progress() * 100))

            logger.info(f"Successfully downloaded file to {destination_path}")
            return True, str(destination_path)

        except HttpError as error:
            logger.error(f"Download failed: {error}")
            return False, f"Download failed: {error}"
        except Exception as e:
            logger.error(f"Unexpected error during download: {str(e)}")
            return False, f"Unexpected error: {str(e)}"

    def download_text_files(
        self,
        text_folder_path: str = GDRIVE_TEXT_FOLDER,
        local_dir: Path = TEXT_DIR,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[List[Dict], str]:
        """
        Download all text files from Investigation_Text folder

        Args:
            text_folder_path: Google Drive folder name
            local_dir: Local directory to save files
            progress_callback: Optional Streamlit progress bar

        Returns:
            Tuple of (downloaded_files: List[Dict], message: str)
        """
        if not self.authenticated:
            logger.warning("Not authenticated - cannot download text files")
            return [], "Not authenticated with Google Drive"

        folder_id = self.get_folder_id(text_folder_path)
        if not folder_id:
            logger.error(f"Folder '{text_folder_path}' not found")
            return [], f"Folder '{text_folder_path}' not found in Google Drive"

        files = self.list_files_in_folder(folder_id)
        downloaded_files = []

        for idx, file in enumerate(files):
            try:
                local_path = local_dir / file['name']
                success, message = self.download_file(file['id'], local_path, progress_callback)

                if success:
                    downloaded_files.append({
                        'id': file['id'],
                        'name': file['name'],
                        'path': str(local_path),
                        'size': file.get('size', 0)
                    })
                    logger.info(f"Downloaded text file: {file['name']}")
                else:
                    logger.warning(f"Failed to download {file['name']}: {message}")

            except Exception as e:
                logger.error(f"Error downloading {file['name']}: {str(e)}")
                continue

        message = f"Downloaded {len(downloaded_files)} text files"
        logger.info(message)
        return downloaded_files, message

    def download_audio_files(
        self,
        audio_folder_path: str = GDRIVE_AUDIO_FOLDER,
        local_dir: Path = AUDIO_DIR,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[List[Dict], str]:
        """
        Download all audio files from Investigation_Audio folder

        Args:
            audio_folder_path: Google Drive folder name
            local_dir: Local directory to save files
            progress_callback: Optional Streamlit progress bar

        Returns:
            Tuple of (downloaded_files: List[Dict], message: str)
        """
        if not self.authenticated:
            logger.warning("Not authenticated - cannot download audio files")
            return [], "Not authenticated with Google Drive"

        folder_id = self.get_folder_id(audio_folder_path)
        if not folder_id:
            logger.error(f"Folder '{audio_folder_path}' not found")
            return [], f"Folder '{audio_folder_path}' not found in Google Drive"

        files = self.list_files_in_folder(folder_id)
        downloaded_files = []

        for idx, file in enumerate(files):
            file_ext = Path(file['name']).suffix.lower()
            
            if file_ext not in SUPPORTED_AUDIO_FORMATS:
                logger.debug(f"Skipping unsupported audio format: {file['name']}")
                continue

            try:
                local_path = local_dir / file['name']
                success, message = self.download_file(file['id'], local_path, progress_callback)

                if success:
                    downloaded_files.append({
                        'id': file['id'],
                        'name': file['name'],
                        'path': str(local_path),
                        'size': file.get('size', 0),
                        'format': file_ext
                    })
                    logger.info(f"Downloaded audio file: {file['name']}")
                else:
                    logger.warning(f"Failed to download {file['name']}: {message}")

            except Exception as e:
                logger.error(f"Error downloading {file['name']}: {str(e)}")
                continue

        message = f"Downloaded {len(downloaded_files)} audio files"
        logger.info(message)
        return downloaded_files, message

    def get_file_metadata(self, file_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific file

        Args:
            file_id: Google Drive file ID

        Returns:
            File metadata dictionary or None if error
        """
        if not self.authenticated:
            logger.warning("Not authenticated - cannot get file metadata")
            return None

        try:
            file = self.service.files().get(
                fileId=file_id,
                fields='id, name, mimeType, createdTime, modifiedTime, size, owners'
            ).execute()

            logger.info(f"Retrieved metadata for file {file_id}")
            return file

        except HttpError as error:
            logger.error(f"Error getting file metadata: {error}")
            return None

    def search_files(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for files in Google Drive

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of file metadata dictionaries
        """
        if not self.authenticated:
            logger.warning("Not authenticated - cannot search files")
            return []

        try:
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, mimeType, createdTime, modifiedTime, size)',
                pageSize=max_results
            ).execute()

            files = results.get('files', [])
            logger.info(f"Search for '{query}' returned {len(files)} results")
            return files

        except HttpError as error:
            logger.error(f"Error searching files: {error}")
            return []

    def get_authentication_status(self) -> Dict[str, any]:
        """
        Get current authentication status and details

        Returns:
            Dictionary with authentication status information
        """
        return {
            'authenticated': self.authenticated,
            'has_credentials': self.creds is not None,
            'credentials_file_exists': self.credentials_path.exists(),
            'token_file_exists': self.token_path.exists(),
            'credentials_path': str(self.credentials_path),
            'token_path': str(self.token_path)
        }


if __name__ == "__main__":
    handler = GDriveHandler()
    success, message = handler.authenticate()
    print(f"Authentication: {message}")
    print(f"Status: {handler.get_authentication_status()}")
