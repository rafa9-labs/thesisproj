"""Machine fingerprinting — generates a stable hardware ID.

Uses WMI/subprocess on Windows to collect CPU, motherboard, BIOS, MAC, and disk serials.
Hashes them together into a 32-char hex fingerprint with tolerance for minor hardware changes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_WINDOWS_WMI_FIELDS = {
    "cpu": r"wmic cpu get ProcessorId",
    "board": r"wmic baseboard get SerialNumber",
    "bios": r"wmic bios get SerialNumber",
    "mac": r"wmic nic where (NetEnabled=true AND NetConnectionStatus=2) get MACAddress",
    "disk": r"wmic diskdrive where Index=0 get SerialNumber",
}


def _run_wmic(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if len(lines) >= 2:
            return lines[1].strip()
        return ""
    except Exception as e:
        logger.debug("wmic command failed: %s → %s", command, e)
        return ""


def _collect_components() -> List[Tuple[str, str]]:
    components: List[Tuple[str, str]] = []
    if platform.system() == "Windows":
        for name, cmd in _WINDOWS_WMI_FIELDS.items():
            val = _run_wmic(cmd)
            if val:
                components.append((name, val.lower().strip()))
    elif platform.system() == "Darwin":
        try:
            sn = subprocess.run(
                ["ioreg", "-l", "-p", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            for line in sn.splitlines():
                if "IOPlatformSerialNumber" in line:
                    val = line.split("=")[-1].strip().strip('"')
                    components.append(("serial", val))
                    break
        except Exception:
            pass
        try:
            uuid = subprocess.run(
                ["sysctl", "-n", "kern.uuid"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if uuid:
                components.append(("uuid", uuid))
        except Exception:
            pass
    else:
        try:
            machine_id = ""
            for path in [
                "/etc/machine-id",
                "/var/lib/dbus/machine-id",
            ]:
                if os.path.exists(path):
                    machine_id = open(path).read().strip()
                    break
            if machine_id:
                components.append(("machine-id", machine_id))
        except Exception:
            pass

    return components


def _fingerprint_from_components(
    components: List[Tuple[str, str]],
    min_match: int = 3,
) -> str:
    raw = "|".join(f"{k}:{v}" for k, v in sorted(components, key=lambda x: x[0]))
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def machine_fingerprint() -> str:
    components = _collect_components()
    if not components:
        hostname = platform.node()
        return hashlib.sha256(f"fallback:{hostname}".encode()).hexdigest()[:32]
    return _fingerprint_from_components(components)


def fingerprint_components() -> List[Tuple[str, str]]:
    return _collect_components()


def matches_stored_fingerprint(
    stored: str,
    current_components: Optional[List[Tuple[str, str]]] = None,
    threshold: float = 0.6,
) -> bool:
    if current_components is None:
        current_components = _collect_components()
    current_hash = _fingerprint_from_components(current_components)
    if current_hash == stored:
        return True
    stored_components = fingerprint_components()
    if not stored_components or not current_components:
        return False
    matching = sum(
        1 for (k1, v1), (_, v2) in zip(
            sorted(stored_components, key=lambda x: x[0]),
            sorted(current_components, key=lambda x: x[0]),
        )
        if v1 == v2 and k1 in dict(current_components)
    )
    total = max(len(stored_components), len(current_components))
    return total > 0 and (matching / total) >= threshold