"""
Google Drive Integration Handler

Handles authentication, file listing, and downloading from Google Drive
for forensic evidence collection
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Generator
import logging
from datetime import datetime

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

from config import (
    GOOGLE_CREDENTIALS_FILE,
    GOOGLE_TOKEN_FILE,
    GOOGLE_SCOPES,
    AUDIO_DIR,
    TEXT_DIR,
    TEMP_DIR
)

logger = logging.getLogger(__name__)

class GoogleDriveHandler:
    """Handles all Google Drive operations for Die Waarheid"""
    
    def __init__(self):
        self.service = None
        self.authenticated = False
        
    def authenticate(self) -> bool:
        """Authenticate with Google Drive using OAuth 2.0"""
        try:
            creds = None
            
            # Check for existing token
            if GOOGLE_TOKEN_FILE.exists():
                with open(GOOGLE_TOKEN_FILE, 'rb') as token:
                    creds = pickle.load(token)
            
            # If there are no (valid) credentials available, let the user log in
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not GOOGLE_CREDENTIALS_FILE.exists():
                        raise FileNotFoundError(
                            f"Google credentials file not found at {GOOGLE_CREDENTIALS_FILE}. "
                            "Please download from Google Cloud Console."
                        )
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        GOOGLE_CREDENTIALS_FILE, GOOGLE_SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save the credentials for the next run
                with open(GOOGLE_TOKEN_FILE, 'wb') as token:
                    pickle.dump(creds, token)
            
            # Build the service
            self.service = build('drive', 'v3', credentials=creds)
            self.authenticated = True
            
            logger.info("Successfully authenticated with Google Drive")
            return True
            
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Drive: {e}")
            return False
    
    def list_files(self, query: str = None, page_size: int = 100) -> List[Dict]:
        """List files from Google Drive with optional query"""
        if not self.authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            # Default query for WhatsApp exports and audio files
            if query is None:
                query = (
                    "(name contains 'WhatsApp' or name contains '.zip' or "
                    "name contains '.opus' or name contains '.m4a' or "
                    "name contains '.mp3' or name contains '.wav') and "
                    "trashed=false"
                )
            
            results = []
            page_token = None
            
            while True:
                response = self.service.files().list(
                    q=query,
                    pageSize=page_size,
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)",
                    pageToken=page_token
                ).execute()
                
                results.extend(response.get('files', []))
                page_token = response.get('nextPageToken')
                
                if not page_token:
                    break
            
            logger.info(f"Found {len(results)} files matching query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def download_file(self, file_id: str, file_name: str, 
                     destination: Path = None) -> Optional[Path]:
        """Download a file from Google Drive"""
        if not self.authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            # Determine destination based on file type
            if destination is None:
                if file_name.endswith(('.zip', '.txt')):
                    destination = TEXT_DIR
                elif file_name.endswith(('.opus', '.m4a', '.mp3', '.wav')):
                    destination = AUDIO_DIR
                else:
                    destination = TEMP_DIR
            
            file_path = destination / file_name
            
            # Download the file
            request = self.service.files().get_media(fileId=file_id)
            
            with io.FileIO(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                
                while done is False:
                    status, done = downloader.next_chunk()
                    logger.debug(f"Download {int(status.progress() * 100)}%")
            
            logger.info(f"Successfully downloaded {file_name} to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download {file_name}: {e}")
            return None
    
    def download_whatsapp_exports(self, output_dir: Path = None) -> List[Path]:
        """Download all WhatsApp export ZIP files from Google Drive"""
        if output_dir is None:
            output_dir = TEXT_DIR / "downloads"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Query for WhatsApp exports
        query = "name contains 'WhatsApp' and name contains '.zip' and trashed=false"
        files = self.list_files(query)
        
        downloaded_files = []
        
        for i, file_info in enumerate(files):
            logger.info(f"Downloading {file_info['name']} ({i+1}/{len(files)})")
            
            file_path = self.download_file(
                file_info['id'],
                file_info['name']
            )
            
            if file_path:
                downloaded_files.append(file_path)
        
        # After downloading all ZIPs, process them together
        if downloaded_files:
            logger.info(f"Downloaded {len(downloaded_files)} WhatsApp ZIP files")
            logger.info("These will be processed together to find the chat file and media")
        
        return downloaded_files
    
    def download_audio_files(self, progress_callback=None) -> List[Path]:
        """Download all audio files"""
        audio_extensions = ['.opus', '.m4a', '.mp3', '.wav']
        query_parts = [f"name contains '{ext}'" for ext in audio_extensions]
        query = f"({' or '.join(query_parts)}) and trashed=false"
        
        files = self.list_files(query)
        
        downloaded_files = []
        
        for i, file_info in enumerate(files):
            if progress_callback:
                progress_callback(i, len(files), f"Downloading {file_info['name']}")
            
            file_path = self.download_file(
                file_info['id'],
                file_info['name']
            )
            
            if file_path:
                downloaded_files.append(file_path)
        
        return downloaded_files
    
    def get_file_info(self, file_id: str) -> Optional[Dict]:
        """Get detailed information about a file"""
        if not self.authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            file_info = self.service.files().get(
                fileId=file_id,
                fields="id, name, mimeType, size, modifiedTime, createdTime, parents, webViewLink"
            ).execute()
            
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return None
    
    def search_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Search for files within a date range"""
        start_str = start_date.isoformat() + 'Z'
        end_str = end_date.isoformat() + 'Z'
        
        query = f"modifiedTime >= '{start_str}' and modifiedTime <= '{end_str}' and trashed=false"
        
        return self.list_files(query)
    
    def create_folder_structure(self, folder_name: str = "Die Waarheid Evidence") -> Optional[str]:
        """Create a folder in Google Drive for organized evidence storage"""
        if not self.authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        try:
            # Check if folder already exists
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            existing = self.list_files(query)
            
            if existing:
                logger.info(f"Folder '{folder_name}' already exists")
                return existing[0]['id']
            
            # Create new folder
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            logger.info(f"Created folder '{folder_name}' with ID: {folder['id']}")
            return folder['id']
            
        except Exception as e:
            logger.error(f"Failed to create folder: {e}")
            return None

# Global instance for easy importing
gdrive_handler = GoogleDriveHandler()
