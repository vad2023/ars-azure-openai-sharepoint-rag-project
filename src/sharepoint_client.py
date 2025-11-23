import logging
from typing import List, Dict, Any

import msal
import requests

from .config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GRAPH_SCOPE = ["https://graph.microsoft.com/.default"]
GRAPH_BASE = "https://graph.microsoft.com/v1.0"


class SharePointClient:
    """Simple Microsoft Graph client for SharePoint document libraries."""

    def __init__(self, cfg: Config = Config):
        cfg.validate()
        self.cfg = cfg
        self._token: str | None = None
        self._session = requests.Session()

    def _get_token(self) -> str:
        if self._token:
            return self._token

        app = msal.ConfidentialClientApplication(
            client_id=self.cfg.SP_CLIENT_ID,
            client_credential=self.cfg.SP_CLIENT_SECRET,
            authority=f"https://login.microsoftonline.com/{self.cfg.TENANT_ID}",
        )

        result = app.acquire_token_for_client(scopes=GRAPH_SCOPE)
        if "access_token" not in result:
            raise RuntimeError(f"Failed to acquire token: {result}")

        self._token = result["access_token"]
        return self._token

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._get_token()}"}

    def get_site(self) -> Dict[str, Any]:
        """Resolve the SharePoint site by hostname + site path."""
        url = f"{GRAPH_BASE}/sites/{self.cfg.SP_SITE_HOSTNAME}:{self.cfg.SP_SITE_PATH}"
        resp = self._session.get(url, headers=self._headers())
        resp.raise_for_status()
        site = resp.json()
        logger.info("Resolved site id=%s", site.get("id"))
        return site

    def get_drive_for_library(self, site_id: str) -> Dict[str, Any]:
        """Find the drive that backs the configured document library."""
        url = f"{GRAPH_BASE}/sites/{site_id}/drives"
        resp = self._session.get(url, headers=self._headers())
        resp.raise_for_status()
        drives = resp.json().get("value", [])
        for d in drives:
            if d.get("name") == self.cfg.SP_DOC_LIBRARY_NAME:
                logger.info(
                    "Found drive for library %s: %s",
                    self.cfg.SP_DOC_LIBRARY_NAME,
                    d.get("id"),
                )
                return d
        raise RuntimeError(f"Library {self.cfg.SP_DOC_LIBRARY_NAME} not found")

    def list_files(self, drive_id: str) -> List[Dict[str, Any]]:
        """List files in the root of the drive (simple demo)."""
        url = f"{GRAPH_BASE}/drives/{drive_id}/root/children"
        resp = self._session.get(url, headers=self._headers())
        resp.raise_for_status()
        items = resp.json().get("value", [])
        files = [i for i in items if "file" in i]
        logger.info("Found %d files in library", len(files))
        return files

    def download_file_content(self, drive_id: str, item_id: str) -> bytes:
        """Download raw file bytes for a given item id."""
        url = f"{GRAPH_BASE}/drives/{drive_id}/items/{item_id}/content"
        resp = self._session.get(url, headers=self._headers(), stream=True)
        resp.raise_for_status()
        return resp.content
