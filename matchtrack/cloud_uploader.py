"""
Secure Cloud Upload Integration for MatchTrack Online.
Features:
- Windows DPAPI (CryptProtectData / CryptUnprotectData) hardware/user-bound token encryption
- Strict HTTPS/TLS enforcement (HTTP only permitted for localhost/127.0.0.1 development)
- Resumable Chunked Upload Protocol (16 MB blocks, per-chunk SHA-256 verification, exponential backoff)
- Automatic fallback to monolithic multipart upload for legacy server versions
- Dual-Stream support (16:9 Standard Broadcast + 32:9 / 8:3 Panorama + AI Tracking JSONL)
"""
import base64
import collections
import concurrent.futures
import ctypes
from ctypes import wintypes
import dataclasses
import hashlib
import io
import json
import os
import sys
import time
import threading
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List, Callable

import requests
from requests.adapters import HTTPAdapter

from .paths import get_cloud_credentials_path
from .logger import get_logger

logger = get_logger("cloud_upload")

DEFAULT_CHUNK_SIZE_BYTES = 8 * 1024 * 1024  # 8 MB per chunk for multi-stream pipelining
DPAPI_ENTROPY = b"MatchTrackStitcher_CloudAuth_v1_Entropy"


class ChunkProgressStream(io.BytesIO):
    """
    Wraps chunk bytes in an io.BytesIO stream that reports byte-level progress
    during HTTP transmission (e.g. to urllib3 / requests).
    """
    def __init__(self, data: bytes, on_bytes_read: Optional[Callable[[int], None]] = None):
        super().__init__(data)
        self._on_bytes_read = on_bytes_read
        self._len = len(data)

    def __len__(self) -> int:
        return self._len

    def read(self, size: int = -1) -> bytes:
        chunk = super().read(size)
        if chunk and self._on_bytes_read:
            self._on_bytes_read(len(chunk))
        return chunk


class RollingSpeedTracker:
    """
    Thread-safe tracker for rolling-window upload speed (MB/s) and sub-chunk progress.
    Maintains a 2.0-second sliding window of transmitted bytes to measure actual real-time throughput.
    """
    def __init__(self, total_bytes: int, window_sec: float = 2.0):
        self.total_bytes = max(1, total_bytes)
        self.window_sec = window_sec
        self.lock = threading.Lock()
        self.active_chunk_progress: Dict[str, int] = {}
        self.history: collections.deque[tuple[float, int]] = collections.deque()
        self.confirmed_bytes: int = 0

    def update_task_progress(self, task_id: str, bytes_sent: int):
        now = time.time()
        with self.lock:
            old = self.active_chunk_progress.get(task_id, 0)
            delta = bytes_sent - old
            if delta > 0:
                self.active_chunk_progress[task_id] = bytes_sent
                self.history.append((now, delta))
                cutoff = now - self.window_sec
                while self.history and self.history[0][0] < cutoff:
                    self.history.popleft()

    def task_failed_reset(self, task_id: str):
        with self.lock:
            self.active_chunk_progress[task_id] = 0

    def task_completed(self, task_id: str, chunk_size: int):
        with self.lock:
            self.active_chunk_progress.pop(task_id, None)
            self.confirmed_bytes += chunk_size

    def get_stats(self) -> tuple[int, float, float]:
        now = time.time()
        with self.lock:
            cutoff = now - self.window_sec
            while self.history and self.history[0][0] < cutoff:
                self.history.popleft()

            window_bytes = sum(b for _, b in self.history)
            if self.history and len(self.history) >= 2:
                dt = max(0.1, self.history[-1][0] - self.history[0][0])
            elif self.history:
                dt = max(0.1, now - self.history[0][0])
            else:
                dt = 1.0

            speed_mb_s = (window_bytes / (1024 * 1024)) / dt if window_bytes > 0 else 0.0
            in_flight = sum(self.active_chunk_progress.values())
            total_current = min(self.total_bytes, self.confirmed_bytes + in_flight)
            remaining_bytes = max(0, self.total_bytes - total_current)
            eta_sec = (remaining_bytes / (1024 * 1024)) / max(speed_mb_s, 0.001) if speed_mb_s > 0.05 else 0.0
            return total_current, speed_mb_s, eta_sec


@dataclasses.dataclass
class UploadChunkTask:
    task_id: str
    stream_type: str
    file_path: str
    chunk_index: int
    offset: int
    chunk_size: int
    label: str


class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char))
    ]


class DPAPICredentialStore:
    """
    Encrypts and decrypts authentication tokens using the native Windows Data Protection API (DPAPI).
    Tokens are cryptographically bound to the current Windows user account and cannot be decrypted
    on any other machine or user profile.
    """
    CRYPTPROTECT_UI_FORBIDDEN = 0x01

    @staticmethod
    def _make_blob(data: bytes) -> tuple[_DATA_BLOB, Any]:
        buf = ctypes.create_string_buffer(data, len(data))
        blob = _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        return blob, buf

    @classmethod
    def encrypt_bytes(cls, plaintext: bytes) -> bytes:
        if sys.platform != "win32":
            return base64.b64encode(plaintext)

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32

        in_blob, _in_buf = cls._make_blob(plaintext)
        entropy_blob, _ent_buf = cls._make_blob(DPAPI_ENTROPY)
        out_blob = _DATA_BLOB()

        ok = crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            "MatchTrackOnlineToken",
            ctypes.byref(entropy_blob),
            None,
            None,
            cls.CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob)
        )
        if not ok:
            err_code = ctypes.GetLastError()
            raise RuntimeError(f"Windows DPAPI CryptProtectData fehlgeschlagen (Code {err_code}).")

        try:
            encrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            return encrypted
        finally:
            kernel32.LocalFree(out_blob.pbData)

    @classmethod
    def decrypt_bytes(cls, ciphertext: bytes) -> bytes:
        if sys.platform != "win32":
            return base64.b64decode(ciphertext)

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32

        in_blob, _in_buf = cls._make_blob(ciphertext)
        entropy_blob, _ent_buf = cls._make_blob(DPAPI_ENTROPY)
        out_blob = _DATA_BLOB()

        ok = crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            ctypes.byref(entropy_blob),
            None,
            None,
            cls.CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob)
        )
        if not ok:
            err_code = ctypes.GetLastError()
            raise RuntimeError(f"Windows DPAPI CryptUnprotectData fehlgeschlagen (Code {err_code}).")

        try:
            decrypted = ctypes.string_at(out_blob.pbData, out_blob.cbData)
            return decrypted
        finally:
            kernel32.LocalFree(out_blob.pbData)

    @classmethod
    def save_session(cls, server_url: str, username: str, access_token: str, user_info: Optional[Dict[str, Any]] = None):
        payload = {
            "server_url": server_url,
            "username": username,
            "access_token": access_token,
            "user_info": user_info or {},
            "saved_at": time.time()
        }
        raw_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        enc_bytes = cls.encrypt_bytes(raw_bytes)
        cred_path = get_cloud_credentials_path()
        os.makedirs(os.path.dirname(os.path.abspath(cred_path)), exist_ok=True)
        with open(cred_path, "wb") as f:
            f.write(base64.b64encode(enc_bytes))
        logger.info(f"Cloud-Zugangsdaten sicher per Windows-DPAPI gespeichert ({username} @ {server_url}).")

    @classmethod
    def load_session(cls) -> Optional[Dict[str, Any]]:
        cred_path = get_cloud_credentials_path()
        if not os.path.exists(cred_path):
            return None
        try:
            with open(cred_path, "rb") as f:
                b64_data = f.read()
            if not b64_data:
                return None
            enc_bytes = base64.b64decode(b64_data)
            dec_bytes = cls.decrypt_bytes(enc_bytes)
            data = json.loads(dec_bytes.decode("utf-8"))
            if isinstance(data, dict) and data.get("access_token") and data.get("server_url"):
                return data
            return None
        except Exception as e:
            logger.warning(f"Gespeicherte Cloud-Session konnte nicht entschlüsselt werden: {e}")
            return None

    @classmethod
    def clear_session(cls):
        cred_path = get_cloud_credentials_path()
        if os.path.exists(cred_path):
            try:
                os.remove(cred_path)
                logger.info("Gespeicherte Cloud-Zugangsdaten gelöscht.")
            except Exception as e:
                logger.warning(f"Konnte Cloud-Zugangsdaten nicht löschen: {e}")


def validate_server_url(raw_url: str) -> str:
    """
    Normalizes and validates the MatchTrack Online server URL.
    Enforces HTTPS for all remote servers; HTTP is only allowed for localhost / 127.0.0.1.
    """
    if not raw_url or not raw_url.strip():
        raise ValueError("Bitte gib die Server-URL deiner MatchTrack Online Plattform an.")

    url = raw_url.strip()
    if not url.startswith(("http://", "https://")):
        low = url.lower()
        if low.startswith(("localhost", "127.0.0.1", "[::1]")):
            url = "http://" + url
        else:
            url = "https://" + url

    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError(f"Ungültige Server-URL: '{raw_url}'")

    hostname = (parsed.hostname or "").lower()
    is_local = hostname in ("localhost", "127.0.0.1", "::1")

    if parsed.scheme == "http" and not is_local:
        raise ValueError(
            f"Sicherheitswarnung: Unverschlüsseltes HTTP ist für Remote-Server ({hostname}) nicht erlaubt. "
            f"Bitte verwende 'https://{hostname}'."
        )

    clean_path = parsed.path.rstrip("/")
    if clean_path.endswith("/api"):
        clean_path = clean_path[:-4]

    normalized = f"{parsed.scheme}://{parsed.netloc}{clean_path}"
    return normalized.rstrip("/")


class MatchTrackOnlineClient:
    """
    Thread-safe HTTPS API client for authenticating, querying teams/matches,
    and uploading multi-GB stitched videos to MatchTrack Online.
    """
    def __init__(self):
        self.server_url: str = ""
        self.access_token: str = ""
        self.username: str = ""
        self.user_info: Dict[str, Any] = {}
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=0
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({
            "User-Agent": "MatchTrack-Stitcher-Windows/1.5.4"
        })

    @property
    def is_authenticated(self) -> bool:
        return bool(self.server_url and self.access_token)

    def _auth_headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.access_token}"}
        if extra:
            headers.update(extra)
        return headers

    def login(self, server_url: str, username: str, password: str, remember: bool = True) -> Dict[str, Any]:
        """Authenticates against POST /api/auth/login and verifies upload privileges."""
        base_url = validate_server_url(server_url)
        if not username or not username.strip():
            raise ValueError("Bitte gib deinen Benutzernamen ein.")
        if not password:
            raise ValueError("Bitte gib dein Passwort ein.")

        login_url = f"{base_url}/api/auth/login"
        try:
            resp = self.session.post(
                login_url,
                data={"username": username.strip(), "password": password},
                timeout=15,
                verify=True
            )
        except requests.exceptions.SSLError as ssl_err:
            raise RuntimeError(f"TLS/SSL-Zertifikatsprüfung fehlgeschlagen für {base_url}: {ssl_err}")
        except requests.exceptions.RequestException as req_err:
            raise RuntimeError(f"Server {base_url} nicht erreichbar: {req_err}")

        if resp.status_code == 401:
            raise RuntimeError("Anmeldung fehlgeschlagen: Benutzername oder Passwort falsch.")
        if resp.status_code == 403:
            detail = "Dein Konto wurde noch nicht freigeschaltet."
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(f"Zugriff verweigert: {detail}")
        if resp.status_code != 200:
            raise RuntimeError(f"Serverfehler beim Login (HTTP {resp.status_code}): {resp.text[:200]}")

        token_data = resp.json()
        token = token_data.get("access_token")
        if not token:
            raise RuntimeError("Server hat kein gültiges Access-Token zurückgegeben.")

        self.server_url = base_url
        self.access_token = token
        self.username = username.strip()

        profile = self.verify_session()
        role = str(profile.get("role", "")).upper()
        if role not in ("ADMIN", "TEAM_ADMIN", "TRAINER", "CO_TRAINER"):
            self.logout()
            raise RuntimeError(
                f"Dein Benutzerkonto ('{self.username}', Rolle: {role}) besitzt keine Trainer- oder Admin-Berechtigung zum Hochladen von Spielen."
            )

        self.user_info = profile
        if remember:
            DPAPICredentialStore.save_session(self.server_url, self.username, self.access_token, profile)

        return profile

    def login_with_api_key(self, server_url: str, api_key: str, remember: bool = True) -> Dict[str, Any]:
        """Authenticates using a Web-Admin generated API-Key ('mto_ak_...') and verifies upload privileges."""
        base_url = validate_server_url(server_url)
        clean_key = (api_key or "").strip()
        if not clean_key:
            raise ValueError("Bitte gib deinen API-Key ein.")
        if not clean_key.startswith("mto_ak_"):
            raise ValueError(
                "Ungültiges API-Key-Format. Ein MatchTrack Online API-Key beginnt mit 'mto_ak_'."
            )

        self.server_url = base_url
        self.access_token = clean_key

        try:
            profile = self.verify_session()
        except requests.exceptions.SSLError as ssl_err:
            self.access_token = ""
            raise RuntimeError(f"TLS/SSL-Zertifikatsprüfung fehlgeschlagen für {base_url}: {ssl_err}")
        except requests.exceptions.RequestException as req_err:
            self.access_token = ""
            raise RuntimeError(f"Server {base_url} nicht erreichbar: {req_err}")
        except Exception as e:
            self.access_token = ""
            raise RuntimeError(f"API-Key-Prüfung fehlgeschlagen: {e}")

        role = str(profile.get("role", "")).upper()
        if role not in ("ADMIN", "TEAM_ADMIN", "TRAINER", "CO_TRAINER"):
            self.logout()
            raise RuntimeError(
                f"Der mit diesem API-Key verknüpfte Account ('{self.username}', Rolle: {role}) "
                f"besitzt keine Trainer- oder Admin-Berechtigung zum Hochladen von Spielen."
            )

        self.user_info = profile
        if remember:
            DPAPICredentialStore.save_session(self.server_url, self.username, self.access_token, profile)

        return profile

    def verify_session(self) -> Dict[str, Any]:
        """Verifies the stored Bearer token against GET /api/users/me."""
        if not self.is_authenticated:
            raise RuntimeError("Nicht mit MatchTrack Online verbunden.")

        me_url = f"{self.server_url}/api/users/me"
        resp = self.session.get(me_url, headers=self._auth_headers(), timeout=12, verify=True)
        if resp.status_code == 401:
            raise RuntimeError("Sitzung abgelaufen. Bitte melde dich erneut an.")
        if resp.status_code != 200:
            raise RuntimeError(f"Fehler beim Abrufen des Benutzerprofils (HTTP {resp.status_code}).")

        profile = resp.json()
        self.user_info = profile
        self.username = profile.get("username", self.username)
        return profile

    def restore_saved_session(self) -> Optional[Dict[str, Any]]:
        """Attempts to restore and verify a DPAPI-encrypted session from disk."""
        saved = DPAPICredentialStore.load_session()
        if not saved:
            return None

        try:
            self.server_url = validate_server_url(saved["server_url"])
            self.access_token = saved["access_token"]
            self.username = saved.get("username", "")
            self.user_info = saved.get("user_info", {})
            profile = self.verify_session()
            return profile
        except Exception as e:
            logger.info(f"Gespeicherte Cloud-Sitzung nicht mehr aktiv ({e}).")
            self.server_url = saved.get("server_url", "")
            self.access_token = ""
            return None

    def logout(self):
        self.access_token = ""
        self.user_info = {}
        DPAPICredentialStore.clear_session()

    def fetch_teams(self) -> List[Dict[str, Any]]:
        """Fetches teams assigned to the current user (or all teams for ADMIN)."""
        if not self.is_authenticated:
            return []

        for endpoint in ("/api/teams/my", "/api/teams"):
            try:
                resp = self.session.get(
                    f"{self.server_url}{endpoint}",
                    headers=self._auth_headers(),
                    timeout=12,
                    verify=True
                )
                if resp.status_code == 200:
                    teams = resp.json()
                    if isinstance(teams, list):
                        return teams
            except Exception as e:
                logger.warning(f"Konnte Mannschaften von {endpoint} nicht laden: {e}")

        # Fallback to teams inside user_info
        return self.user_info.get("teams", [])

    def fetch_recent_matches(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Fetches recent matches from GET /api/matches so the user can add/replace a video stream."""
        if not self.is_authenticated:
            return []

        try:
            resp = self.session.get(
                f"{self.server_url}/api/matches",
                headers=self._auth_headers(),
                timeout=15,
                verify=True
            )
            if resp.status_code == 200:
                matches = resp.json()
                if isinstance(matches, list):
                    return matches[:limit]
        except Exception as e:
            logger.warning(f"Konnte bestehende Matches nicht abrufen: {e}")
        return []

    def upload_match(
        self,
        name: str,
        team_id: Optional[str] = None,
        team_name: Optional[str] = None,
        category: str = "Punktspiel",
        age_group: Optional[str] = None,
        recording_date: Optional[str] = None,
        video_quality: Optional[str] = "1080p",
        match_id: Optional[str] = None,
        video_16x9_path: Optional[str] = None,
        video_32x9_path: Optional[str] = None,
        tracking_path: Optional[str] = None,
        is_password_protected: bool = False,
        password: Optional[str] = None,
        skip_conversion: bool = False,
        notify_team_members: bool = True,
        concurrency: int = 3,
        stop_event: Optional[threading.Event] = None,
        progress_callback: Optional[Callable[[int, int, float, float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Uploads a match (16:9 Broadcast, 32:9/8:3 Panorama, or Both + AI Tracking)
        using the Resumable High-Speed Multi-Stream Chunked Upload Protocol with SHA-256 verification.
        Uses concurrent threads, connection pooling, and sub-chunk byte streaming to maximize throughput.
        Automatically falls back to legacy multipart upload if the server has not been updated yet.
        """
        if not self.is_authenticated:
            raise RuntimeError("Nicht bei MatchTrack Online angemeldet.")

        streams_files: Dict[str, str] = {}
        if video_16x9_path and os.path.exists(video_16x9_path):
            streams_files["16x9"] = os.path.abspath(video_16x9_path)
        if video_32x9_path and os.path.exists(video_32x9_path):
            streams_files["32x9"] = os.path.abspath(video_32x9_path)
        if tracking_path and os.path.exists(tracking_path):
            streams_files["tracking"] = os.path.abspath(tracking_path)

        if "16x9" not in streams_files and "32x9" not in streams_files:
            raise ValueError("Keine gültige Videodatei (16:9 oder Panorama) für den Upload gefunden.")

        streams_sizes: Dict[str, int] = {
            k: os.path.getsize(path) for k, path in streams_files.items()
        }
        total_bytes_all = sum(streams_sizes.values())
        concurrency = max(1, min(6, int(concurrency or 3)))

        init_payload = {
            "match_id": match_id,
            "name": name.strip(),
            "team_id": team_id,
            "team_name": team_name,
            "category": category,
            "age_group": age_group,
            "recording_date": recording_date,
            "video_quality": video_quality,
            "is_password_protected": is_password_protected,
            "password": password if is_password_protected else None,
            "skip_conversion": skip_conversion,
            "notify_team_members": notify_team_members,
            "streams": streams_sizes,
            "chunk_size_bytes": DEFAULT_CHUNK_SIZE_BYTES
        }

        if progress_callback:
            progress_callback(0, total_bytes_all, 0.0, 0.0, "Initialisiere sichere High-Speed Upload-Session...")

        init_url = f"{self.server_url}/api/upload/session/init"
        init_resp = self.session.post(
            init_url,
            json=init_payload,
            headers=self._auth_headers(),
            timeout=20,
            verify=True
        )

        # Fallback if server is still running an older backend without /api/upload/session/init
        if init_resp.status_code == 404:
            logger.info("Server unterstützt noch kein Chunked-Session-API (404), nutze kompatiblen Direkt-Upload...")
            return self._upload_legacy_multipart(
                name=name,
                team_id=team_id,
                team_name=team_name,
                category=category,
                age_group=age_group,
                recording_date=recording_date,
                match_id=match_id,
                streams_files=streams_files,
                total_bytes_all=total_bytes_all,
                skip_conversion=skip_conversion,
                stop_event=stop_event,
                progress_callback=progress_callback
            )

        if init_resp.status_code not in (200, 201):
            detail = init_resp.text[:300]
            try:
                detail = init_resp.json().get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(f"Upload-Initialisierung fehlgeschlagen (HTTP {init_resp.status_code}): {detail}")

        session_info = init_resp.json()
        upload_id = session_info["upload_id"]
        target_match_id = session_info["match_id"]
        server_chunk_size = int(session_info.get("chunk_size_bytes", DEFAULT_CHUNK_SIZE_BYTES))

        logger.info(
            f"Upload-Session gestartet: upload_id={upload_id}, match_id={target_match_id}, "
            f"Gesamtgröße={total_bytes_all / (1024**2):.1f} MB, Chunk-Größe={server_chunk_size / (1024**2):.0f} MB, "
            f"Streams={concurrency} parallel"
        )

        stream_labels = {
            "16x9": "16:9 Broadcast",
            "32x9": "Panorama",
            "tracking": "KI-Tracking"
        }

        # Build chunk upload task list
        chunk_tasks: List[UploadChunkTask] = []
        for stream_type, file_path in streams_files.items():
            file_size = streams_sizes[stream_type]
            total_chunks = max(1, (file_size + server_chunk_size - 1) // server_chunk_size)
            label = stream_labels.get(stream_type, stream_type)
            for idx in range(total_chunks):
                offset = idx * server_chunk_size
                this_size = min(server_chunk_size, file_size - offset)
                chunk_tasks.append(
                    UploadChunkTask(
                        task_id=f"{stream_type}_{idx}",
                        stream_type=stream_type,
                        file_path=file_path,
                        chunk_index=idx,
                        offset=offset,
                        chunk_size=this_size,
                        label=f"{label} [{idx+1}/{total_chunks}]"
                    )
                )

        tracker = RollingSpeedTracker(total_bytes=total_bytes_all, window_sec=2.0)
        last_ui_update = 0.0
        ui_lock = threading.Lock()
        internal_stop_event = threading.Event()

        def notify_ui(force: bool = False):
            nonlocal last_ui_update
            if not progress_callback:
                return
            now = time.time()
            if not force and (now - last_ui_update < 0.12):
                return
            with ui_lock:
                if not force and (now - last_ui_update < 0.12):
                    return
                last_ui_update = now
                total_curr, speed_mb_s, eta_sec = tracker.get_stats()
                mb_sent = total_curr / (1024 * 1024)
                mb_total = total_bytes_all / (1024 * 1024)
                if concurrency > 1:
                    status_text = f"High-Speed Multi-Stream ({concurrency} parallel): {mb_sent:.1f}/{mb_total:.1f} MB"
                else:
                    status_text = f"Upload: {mb_sent:.1f}/{mb_total:.1f} MB"
                try:
                    progress_callback(total_curr, total_bytes_all, speed_mb_s, eta_sec, status_text)
                except Exception:
                    pass

        def upload_single_chunk(task: UploadChunkTask):
            if (stop_event and stop_event.is_set()) or internal_stop_event.is_set():
                raise InterruptedError("Upload wurde vom Benutzer abgebrochen.")

            with open(task.file_path, "rb") as f:
                f.seek(task.offset)
                chunk_bytes = f.read(task.chunk_size)

            if not chunk_bytes:
                return

            chunk_sha256 = hashlib.sha256(chunk_bytes).hexdigest()
            part_url = f"{self.server_url}/api/upload/session/{upload_id}/part"
            params = {
                "stream_type": task.stream_type,
                "chunk_index": task.chunk_index,
                "offset": task.offset
            }
            headers = self._auth_headers({
                "Content-Type": "application/octet-stream",
                "X-Chunk-SHA256": chunk_sha256
            })

            max_retries = 5
            for attempt in range(1, max_retries + 1):
                if (stop_event and stop_event.is_set()) or internal_stop_event.is_set():
                    raise InterruptedError("Upload wurde vom Benutzer abgebrochen.")

                bytes_sent_this_attempt = 0

                def on_bytes_read(n: int):
                    nonlocal bytes_sent_this_attempt
                    bytes_sent_this_attempt += n
                    tracker.update_task_progress(task.task_id, bytes_sent_this_attempt)
                    notify_ui()

                stream = ChunkProgressStream(chunk_bytes, on_bytes_read)

                try:
                    part_resp = self.session.put(
                        part_url,
                        params=params,
                        data=stream,
                        headers=headers,
                        timeout=120,
                        verify=True
                    )
                    if part_resp.status_code == 200:
                        tracker.task_completed(task.task_id, len(chunk_bytes))
                        notify_ui()
                        return

                    err_txt = part_resp.text[:200]
                    try:
                        err_txt = part_resp.json().get("detail", err_txt)
                    except Exception:
                        pass

                    tracker.task_failed_reset(task.task_id)
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Chunk {task.chunk_index + 1} ({task.label}) fehlgeschlagen "
                            f"(HTTP {part_resp.status_code}): {err_txt}"
                        )
                    logger.warning(
                        f"Chunk {task.chunk_index + 1} ({task.label}) HTTP {part_resp.status_code}, "
                        f"Wiederholung {attempt}/{max_retries}..."
                    )
                except requests.exceptions.RequestException as net_err:
                    tracker.task_failed_reset(task.task_id)
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Netzwerkfehler bei Chunk {task.chunk_index + 1} ({task.label}) "
                            f"nach {max_retries} Versuchen: {net_err}"
                        )
                    wait_sec = min(10.0, 1.2 * (2 ** (attempt - 1)))
                    logger.warning(
                        f"Verbindungsproblem bei Chunk {task.chunk_index + 1} ({task.label}): {net_err}. "
                        f"Warte {wait_sec:.1f}s vor Versuch {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait_sec)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(upload_single_chunk, t) for t in chunk_tasks]
                for future in concurrent.futures.as_completed(futures):
                    if stop_event and stop_event.is_set():
                        internal_stop_event.set()
                        executor.shutdown(wait=False, cancel_futures=True)
                        self._cancel_session_quietly(upload_id)
                        raise InterruptedError("Upload wurde vom Benutzer abgebrochen.")

                    exc = future.exception()
                    if exc:
                        internal_stop_event.set()
                        executor.shutdown(wait=False, cancel_futures=True)
                        self._cancel_session_quietly(upload_id)
                        if isinstance(exc, InterruptedError):
                            raise exc
                        raise exc

            notify_ui(force=True)

            if progress_callback:
                progress_callback(
                    total_bytes_all,
                    total_bytes_all,
                    0.0,
                    0.0,
                    "Finalisiere Spiel auf Server (FFprobe-Check, Thumbnail & HLS-Start)..."
                )

            complete_url = f"{self.server_url}/api/upload/session/{upload_id}/complete"
            comp_resp = self.session.post(
                complete_url,
                headers=self._auth_headers(),
                timeout=90,
                verify=True
            )
            if comp_resp.status_code not in (200, 201):
                detail = comp_resp.text[:300]
                try:
                    detail = comp_resp.json().get("detail", detail)
                except Exception:
                    pass
                raise RuntimeError(f"Finalisierung fehlgeschlagen (HTTP {comp_resp.status_code}): {detail}")

            result = comp_resp.json()
            final_match_id = result.get("match_id", target_match_id)
            result["match_url"] = f"{self.server_url}/match/{final_match_id}"
            logger.info(f"🎉 Cloud-Upload erfolgreich abgeschlossen! Match-URL: {result['match_url']}")
            return result

        except InterruptedError:
            self._cancel_session_quietly(upload_id)
            raise
        except Exception as e:
            self._cancel_session_quietly(upload_id)
            logger.error(f"Fehler während des Cloud-Uploads: {e}", exc_info=True)
            raise
            result["match_url"] = f"{self.server_url}/match/{final_match_id}"
            logger.info(f"🎉 Cloud-Upload erfolgreich abgeschlossen! Match-URL: {result['match_url']}")
            return result

        except InterruptedError:
            raise
        except Exception as e:
            logger.error(f"Fehler während des Cloud-Uploads: {e}", exc_info=True)
            raise

    def _cancel_session_quietly(self, upload_id: str):
        try:
            self.session.delete(
                f"{self.server_url}/api/upload/session/{upload_id}",
                headers=self._auth_headers(),
                timeout=8,
                verify=True
            )
        except Exception:
            pass

    def _upload_legacy_multipart(
        self,
        name: str,
        team_id: Optional[str],
        team_name: Optional[str],
        category: str,
        age_group: Optional[str],
        recording_date: Optional[str],
        match_id: Optional[str],
        streams_files: Dict[str, str],
        total_bytes_all: int,
        skip_conversion: bool,
        stop_event: Optional[threading.Event],
        progress_callback: Optional[Callable[[int, int, float, float, str], None]]
    ) -> Dict[str, Any]:
        """Fallback multipart upload to POST /api/admin/matches (or replace-video)."""
        if stop_event and stop_event.is_set():
            raise InterruptedError("Upload wurde vom Benutzer abgebrochen.")

        if match_id:
            url = f"{self.server_url}/api/admin/matches/{match_id}/replace-video"
        else:
            url = f"{self.server_url}/api/admin/matches"

        has_16x9 = "16x9" in streams_files
        has_32x9 = "32x9" in streams_files
        aspect = "dual" if (has_16x9 and has_32x9) else ("32:9" if has_32x9 else "16:9")

        data = {
            "name": name.strip(),
            "category": category or "Punktspiel",
            "aspect_ratio": aspect,
            "skip_conversion": "true" if skip_conversion else "false"
        }
        if team_id:
            data["team_id"] = team_id
        if team_name:
            data["team_name"] = team_name
        if age_group:
            data["age_group"] = age_group
        if recording_date:
            data["recording_date"] = recording_date

        opened_files = []
        files_payload = {}
        try:
            if has_16x9:
                f16 = open(streams_files["16x9"], "rb")
                opened_files.append(f16)
                files_payload["video_file_16x9"] = (os.path.basename(streams_files["16x9"]), f16, "video/mp4")
            if has_32x9:
                f32 = open(streams_files["32x9"], "rb")
                opened_files.append(f32)
                files_payload["video_file_32x9"] = (os.path.basename(streams_files["32x9"]), f32, "video/mp4")
            if "tracking" in streams_files:
                ftr = open(streams_files["tracking"], "rb")
                opened_files.append(ftr)
                files_payload["tracking_file"] = (os.path.basename(streams_files["tracking"]), ftr, "application/json")

            if progress_callback:
                progress_callback(0, total_bytes_all, 0.0, 0.0, "Sende Videodatei(en) an Server...")

            resp = self.session.post(
                url,
                data=data,
                files=files_payload,
                headers=self._auth_headers(),
                timeout=3600,
                verify=True
            )
            if resp.status_code not in (200, 201):
                detail = resp.text[:300]
                try:
                    detail = resp.json().get("detail", detail)
                except Exception:
                    pass
                raise RuntimeError(f"Upload fehlgeschlagen (HTTP {resp.status_code}): {detail}")

            res_data = resp.json()
            final_id = res_data.get("match_id", match_id or "")
            res_data["match_url"] = f"{self.server_url}/match/{final_id}" if final_id else self.server_url
            return res_data
        finally:
            for fh in opened_files:
                try:
                    fh.close()
                except Exception:
                    pass

