"""Integración aislada con EcoSort para el Smart Recycling Bin."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from dotenv import load_dotenv


# Carga la configuración local sin sobrescribir variables definidas por el sistema.
load_dotenv(Path(__file__).with_name(".env"))


MATERIALES_PERMITIDOS = {"glass", "plastic"}


class EcoSortError(Exception):
    """Error controlado de configuración, red o respuesta de EcoSort."""


def _bool_entorno(nombre, defecto=False):
    valor = os.environ.get(nombre)
    if valor is None:
        return defecto
    return valor.strip().lower() in {"1", "true", "yes", "on"}


def _fecha_iso(valor):
    if not isinstance(valor, str):
        raise EcoSortError("EcoSort respondió expires_at inválido.")
    try:
        fecha = datetime.fromisoformat(valor.replace("Z", "+00:00"))
    except ValueError as error:
        raise EcoSortError("EcoSort respondió expires_at inválido.") from error
    if fecha.tzinfo is None:
        raise EcoSortError("EcoSort respondió expires_at sin zona horaria.")
    return fecha.astimezone(timezone.utc)


class EcoSortClient:
    """Cliente de una sola solicitud, deliberadamente sin reintentos."""

    def __init__(self):
        self.enabled = _bool_entorno("ECOSORT_QR_ENABLED", False)
        self.base_url = os.environ.get("ECOSORT_API_BASE_URL", "").rstrip("/")
        self.device_key = os.environ.get("ECOSORT_DEVICE_API_KEY", "")
        try:
            self.display_seconds = max(
                0.0, float(os.environ.get("ECOSORT_QR_DISPLAY_SECONDS", "120"))
            )
        except ValueError:
            self.display_seconds = 120.0

    def solicitar_reciclaje(self, material, confianza):
        """Hace un único POST y devuelve la respuesta validada de EcoSort."""
        if material not in MATERIALES_PERMITIDOS:
            raise EcoSortError("Material no permitido para EcoSort.")
        if not self.enabled:
            return None
        if not self.base_url or not self.device_key:
            raise EcoSortError("EcoSort no está configurado; QR omitido.")
        if not self.base_url.startswith("https://"):
            raise EcoSortError("ECOSORT_API_BASE_URL debe usar HTTPS.")

        datos = json.dumps({"material": material, "confidence": float(confianza)}).encode("utf-8")
        solicitud = Request(
            f"{self.base_url}/api/recycle",
            data=datos,
            headers={"Content-Type": "application/json", "x-device-key": self.device_key},
            method="POST",
        )
        try:
            with urlopen(solicitud, timeout=8) as respuesta:
                cuerpo = json.loads(respuesta.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            # No se reintenta: la recepción del servidor puede ser incierta.
            raise EcoSortError("No se pudo contactar EcoSort; QR omitido.") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EcoSortError("EcoSort devolvió una respuesta inválida; QR omitido.") from error

        if not isinstance(cuerpo, dict) or cuerpo.get("accepted") is not True:
            raise EcoSortError("EcoSort rechazó el reciclaje; QR omitido.")
        claim_url = cuerpo.get("claim_url")
        if not isinstance(claim_url, str) or not claim_url.startswith("https://"):
            raise EcoSortError("EcoSort devolvió claim_url inválida; QR omitido.")

        return {"claim_url": claim_url, "expires_at": _fecha_iso(cuerpo.get("expires_at"))}

    def segundos_visibles(self, expires_at, ahora=None):
        """Aplica el menor límite entre configuración local y expiración remota."""
        ahora = ahora or datetime.now(timezone.utc)
        return max(0.0, min(self.display_seconds, (expires_at - ahora).total_seconds()))

    @staticmethod
    def crear_imagen_qr(claim_url):
        """Genera el QR sólo en memoria, a partir de la URL autorizada."""
        try:
            import qrcode
        except ImportError as error:
            raise EcoSortError("Falta la dependencia qrcode para mostrar el QR.") from error
        return np.array(qrcode.make(claim_url).convert("RGB"))
