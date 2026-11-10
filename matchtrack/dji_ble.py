"""
DJI Osmo Action 4 Bluetooth Low Energy (BLE) Controller & Stream Initializer.
Bypasses the DJI Mimo mobile app by connecting directly to DJI Action cameras over BLE (WinRT on Windows)
and dispatching DUML-framed Wi-Fi connection and RTMP live-streaming commands.
Also provides fallback Live-Stream QR-Code generation for instant lens-scanning.
"""
import asyncio
import io
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable, Any

try:
    from bleak import BleakScanner, BleakClient
    from bleak.backends.device import BLEDevice
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    BleakScanner = None
    BleakClient = None
    BLEDevice = None

from .logger import get_logger

logger = get_logger("dji_ble")

# DJI GATT Service & Characteristic UUIDs
DJI_SERVICE_UUID = "0000fe00-0000-1000-8000-00805f9b34fb"
DJI_BASE_SERVICE_UUID = "0000fff0-0000-1000-8000-00805f9b34fb"
DJI_WRITE_CHAR_UUID = "0000fff5-0000-1000-8000-00805f9b34fb"    # Write without response
DJI_NOTIFY_CHAR_UUID = "0000fff4-0000-1000-8000-00805f9b34fb"   # Notifications / responses


def calc_dji_crc8(data: bytes, seed: int = 0x77) -> int:
    """Calculates DJI DUML 8-bit header CRC."""
    crc = seed
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x01:
                crc = (crc >> 1) ^ 0x8C
            else:
                crc >>= 1
    return crc & 0xFF


def calc_dji_crc16(data: bytes, seed: int = 0x3692) -> int:
    """Calculates DJI DUML 16-bit packet CRC."""
    crc = seed
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return crc & 0xFFFF


def build_duml_packet(
    cmd_set: int,
    cmd_id: int,
    payload: bytes,
    seq: int = 1,
    sender: int = 0x01,  # Mobile App / PC Controller
    receiver: int = 0x02  # Camera Subsystem
) -> bytes:
    """
    Constructs a complete DUML protocol packet with magic 0x55, CRC8 header,
    subsystem routing, command set/id, payload, and CRC16 footer.
    """
    # Packet length = Header(4) + Routing/Seq(6) + Payload(len) + CRC16(2)
    pkt_len = 4 + 6 + len(payload) + 2
    ver = 0x01  # DUML v1

    # Header: Magic(0x55) | Length(10 bits) + Version(6 bits)
    # Byte 0: 0x55
    # Byte 1: length LSB (bits 0-7)
    # Byte 2: length MSB (bits 0-1) | (ver << 2)
    b0 = 0x55
    b1 = pkt_len & 0xFF
    b2 = ((pkt_len >> 8) & 0x03) | ((ver & 0x3F) << 2)

    header_crc = calc_dji_crc8(bytes([b0, b1, b2]))
    header = bytes([b0, b1, b2, header_crc])

    # Routing & Sequence
    routing = bytes([
        sender & 0xFF,
        receiver & 0xFF,
        seq & 0xFF,
        (seq >> 8) & 0xFF,
        cmd_set & 0xFF,
        cmd_id & 0xFF
    ])

    body = routing + payload
    full_before_crc16 = header + body
    crc16_val = calc_dji_crc16(full_before_crc16)
    footer = struct.pack("<H", crc16_val)

    return full_before_crc16 + footer


@dataclass
class DjiCameraDevice:
    name: str
    address: str
    rssi: int
    is_action4: bool = False
    connected: bool = False
    streaming: bool = False
    battery_pct: int = -1


class DjiBLEManager:
    """
    Manages Bluetooth Low Energy discovery, pairing, and live-streaming commands
    for DJI Osmo Action cameras from a Windows PC.
    """
    def __init__(self):
        self.discovered_devices: Dict[str, DjiCameraDevice] = {}
        self.clients: Dict[str, Any] = {}
        self.is_scanning = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Callbacks for GUI
        self.on_device_found: Optional[Callable[[DjiCameraDevice], None]] = None
        self.on_connection_change: Optional[Callable[[str, bool], None]] = None
        self.on_stream_started: Optional[Callable[[str, str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        self._ensure_event_loop()

    def _ensure_event_loop(self):
        if self._worker_thread is None or not self._worker_thread.is_alive():
            ready_event = threading.Event()

            def run_loop():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                ready_event.set()
                self._loop.run_forever()

            self._worker_thread = threading.Thread(target=run_loop, daemon=True, name="DjiBLEWorker")
            self._worker_thread.start()
            ready_event.wait(timeout=2.0)

    def scan_for_cameras(self, timeout_sec: float = 6.0):
        """Scans for nearby DJI Osmo Action cameras asynchronously."""
        if not BLEAK_AVAILABLE or self._loop is None:
            logger.warning("Bleak ist nicht verfügbar. Bluetooth BLE kann nicht gestartet werden.")
            return

        asyncio.run_coroutine_threadsafe(self._async_scan(timeout_sec), self._loop)

    async def _async_scan(self, timeout_sec: float):
        self.is_scanning = True
        logger.info("Starte BLE-Suche nach DJI Osmo Action Kameras...")
        try:
            devices = await BleakScanner.discover(timeout=timeout_sec, return_adv=True)
            for d, adv in devices.values():
                name = d.name or adv.local_name or "Unbekannt"
                low_name = name.lower()
                is_dji = any(k in low_name for k in ["action", "osmo", "dji"])

                # Check DJI service UUIDs in advertisement data
                service_uuids = [str(u).lower() for u in (adv.service_uuids or [])]
                if "0000fe00" in " ".join(service_uuids) or "0000fff0" in " ".join(service_uuids):
                    is_dji = True

                if is_dji:
                    is_a4 = "action 4" in low_name or "action4" in low_name
                    cam_dev = DjiCameraDevice(
                        name=name,
                        address=d.address,
                        rssi=adv.rssi,
                        is_action4=is_a4
                    )
                    with self._lock:
                        self.discovered_devices[d.address] = cam_dev
                    logger.info(f"DJI Kamera entdeckt: {name} ({d.address}, RSSI: {adv.rssi})")
                    if self.on_device_found:
                        self.on_device_found(cam_dev)
        except Exception as e:
            logger.warning(f"Fehler bei BLE-Suche: {e}")
            if self.on_error:
                self.on_error(f"Bluetooth-Scan fehlgeschlagen: {e}")
        finally:
            self.is_scanning = False

    def connect_camera(self, address: str):
        """Connects and pairs with a discovered camera via BLE."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._async_connect(address), self._loop)

    async def _async_connect(self, address: str):
        logger.info(f"Verbinde per BLE mit Kamera {address}...")
        try:
            client = BleakClient(address)
            await client.connect(timeout=15.0)
            if client.is_connected:
                with self._lock:
                    self.clients[address] = client
                    if address in self.discovered_devices:
                        self.discovered_devices[address].connected = True
                logger.info(f"Erfolgreich mit DJI Kamera {address} verbunden!")
                if self.on_connection_change:
                    self.on_connection_change(address, True)
            else:
                raise RuntimeError("Verbindung konnte nicht hergestellt werden.")
        except Exception as e:
            logger.error(f"BLE-Verbindungsfehler bei {address}: {e}")
            if self.on_error:
                self.on_error(f"Konnte Kamera {address} nicht verbinden: {e}")

    def disconnect_camera(self, address: str):
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._async_disconnect(address), self._loop)

    async def _async_disconnect(self, address: str):
        client = self.clients.pop(address, None)
        if client:
            try:
                await client.disconnect()
            except Exception:
                pass
        with self._lock:
            if address in self.discovered_devices:
                self.discovered_devices[address].connected = False
                self.discovered_devices[address].streaming = False
        if self.on_connection_change:
            self.on_connection_change(address, False)

    def start_rtmp_streaming(
        self,
        address: str,
        wifi_ssid: str,
        wifi_password: str,
        rtmp_url: str,
        bitrate_kbps: int = 6000
    ):
        """
        Sends the DUML payload containing Wi-Fi credentials and RTMP ingest URL
        to trigger camera-side live streaming over Wi-Fi without the DJI Mimo app.
        """
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._async_start_rtmp(address, wifi_ssid, wifi_password, rtmp_url, bitrate_kbps),
                self._loop
            )

    async def _async_start_rtmp(
        self,
        address: str,
        wifi_ssid: str,
        wifi_password: str,
        rtmp_url: str,
        bitrate_kbps: int
    ):
        client = self.clients.get(address)
        if not client or not client.is_connected:
            err = f"Kamera {address} ist nicht per Bluetooth verbunden."
            logger.warning(err)
            if self.on_error:
                self.on_error(err)
            return

        logger.info(f"Sende RTMP-Startbefehl an Kamera {address} (Ziel: {rtmp_url}, WLAN: {wifi_ssid})...")

        # Encode Wi-Fi & RTMP payload for DJI Action 4
        # Format: [ssid_bytes + \0] + [pw_bytes + \0] + [rtmp_bytes + \0] + [resolution=1 (1080p)] + [bitrate_kbps uint32]
        payload = (
            wifi_ssid.encode("utf-8") + b"\x00" +
            wifi_password.encode("utf-8") + b"\x00" +
            rtmp_url.encode("utf-8") + b"\x00" +
            struct.pack("<B", 0x01) +  # 0x01 = 1080p, 0x02 = 720p
            struct.pack("<I", int(bitrate_kbps))
        )

        # DUML Command: CmdSet = 0x02 (Camera), CmdID = 0xD2 (Start Live Stream)
        packet = build_duml_packet(cmd_set=0x02, cmd_id=0xD2, payload=payload, seq=101)

        try:
            # Write to characteristic FFF5 (writeWithoutResponse or writeWithResponse)
            try:
                await client.write_gatt_char(DJI_WRITE_CHAR_UUID, packet, response=False)
            except Exception:
                await client.write_gatt_char(DJI_WRITE_CHAR_UUID, packet, response=True)

            logger.info(f"🎉 RTMP-Startbefehl erfolgreich an Kamera {address} übermittelt!")
            with self._lock:
                if address in self.discovered_devices:
                    self.discovered_devices[address].streaming = True

            if self.on_stream_started:
                self.on_stream_started(address, rtmp_url)

        except Exception as e:
            logger.error(f"Fehler beim Senden des RTMP-Befehls an {address}: {e}")
            if self.on_error:
                self.on_error(f"Stream-Start auf Kamera fehlgeschlagen: {e}")

    @staticmethod
    def generate_dji_stream_qr_image(
        wifi_ssid: str,
        wifi_password: str,
        rtmp_url: str,
        bitrate_kbps: int = 6000
    ) -> Optional[Any]:
        """
        Generates a DJI Action compatible QR-Code image that can be scanned by the camera lens
        to instantly initiate Wi-Fi RTMP streaming without typing or Bluetooth pairing.
        """
        try:
            import qrcode
            # Standard DJI Livestream QR format
            qr_content = f"dji_stream://v1?ssid={wifi_ssid}&pwd={wifi_password}&url={rtmp_url}&bps={bitrate_kbps}"
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=8,
                border=3
            )
            qr.add_data(qr_content)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            return img
        except Exception as e:
            logger.warning(f"Konnte QR-Code nicht generieren: {e}")
            return None

