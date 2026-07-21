"""voz.py - Salida de voz offline para el Smart Bin.

Habla texto dinámico (cualquier frase) SIN internet. Elige el mejor motor
disponible en tiempo de ejecución:

  1. Piper  -> voz neuronal natural en español (recomendado).
  2. pyttsx3 -> usa espeak-ng, voz más robótica pero se instala con un pip.

Si ningún motor funciona (o no hay audio), degrada a sólo imprimir la frase
en consola: la voz NUNCA debe tumbar el bucle principal del bin.

La síntesis y la reproducción corren en un HILO aparte para no congelar el
bucle de la cámara, y hay un enfriamiento (cooldown) para no repetir la misma
frase demasiado seguido.

--- Instalación ---
Opción sencilla (robótica):
    pip install pyttsx3
    sudo apt install espeak-ng          # Linux/WSL

Opción con voz natural (Piper):
    pip install piper-tts
    # descarga una voz en español, p. ej.:
    #   es_MX-claude-high  o  es_ES-davefx-medium  (archivos .onnx + .onnx.json)
    # y exporta la ruta del modelo:
    export PIPER_VOICE=/ruta/a/es_MX-claude-high.onnx
    # (opcional) export PIPER_BIN=/ruta/al/binario/piper
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time


def _buscar_reproductor():
    """Devuelve un comando para reproducir un WAV, o None si no hay ninguno."""
    for nombre in ("paplay", "aplay", "ffplay"):
        ruta = shutil.which(nombre)
        if ruta:
            if nombre == "ffplay":
                return [ruta, "-nodisp", "-autoexit", "-loglevel", "quiet"]
            return [ruta]
    return None


class Locutor:
    """Reproduce frases por voz de forma no bloqueante y con enfriamiento."""

    def __init__(self, cooldown=8.0, piper_bin=None, piper_voice=None):
        self.cooldown = float(cooldown)
        self._ultimo = 0.0
        self._hablando = threading.Event()  # evita solapar dos frases
        self.motor = None                    # "piper" | "pyttsx3" | None
        # Mensaje corto para pintar en la ventana cuando el audio falla o no hay
        # motor. None = todo bien. El bin sigue funcionando pase lo que pase.
        self.mensaje_ui = None

        # --- Intento 1: Piper (voz natural) ---
        piper_bin = piper_bin or os.environ.get("PIPER_BIN") or shutil.which("piper")
        piper_voice = piper_voice or os.environ.get("PIPER_VOICE")
        self._piper_bin = piper_bin
        self._piper_voice = piper_voice
        self._reproductor = _buscar_reproductor()
        if piper_bin and piper_voice and os.path.exists(piper_voice) and self._reproductor:
            self.motor = "piper"

        # --- Intento 2: pyttsx3 (robótico pero fácil) ---
        if self.motor is None:
            try:
                import pyttsx3  # noqa: F401
                self.motor = "pyttsx3"
            except Exception:
                self.motor = None

        if self.motor:
            print(f"🔊 Voz activada (motor: {self.motor}).")
        else:
            self.mensaje_ui = "Voz no disponible (sin motor TTS) - bin operativo"
            print(
                "🔇 Sin motor de voz. Instala 'pyttsx3' (+ espeak-ng) o configura "
                "Piper. Por ahora sólo se imprime el mensaje."
            )

    # ------------------------------------------------------------------
    def decir(self, texto, forzar=False):
        """Habla 'texto' en segundo plano. Respeta el cooldown salvo forzar=True."""
        ahora = time.time()
        if not forzar and (ahora - self._ultimo) < self.cooldown:
            return  # aún en enfriamiento
        if self._hablando.is_set():
            return  # ya hay una frase sonando; no encimamos
        self._ultimo = ahora
        self._hablando.set()
        hilo = threading.Thread(target=self._trabajo, args=(texto,), daemon=True)
        hilo.start()

    # ------------------------------------------------------------------
    def _trabajo(self, texto):
        try:
            if self.motor == "piper":
                self._decir_piper(texto)
            elif self.motor == "pyttsx3":
                self._decir_pyttsx3(texto)
            else:
                print(f"🗣️  (voz) {texto}")
                return
            # Sonó bien: si había un aviso de fallo transitorio, lo limpiamos.
            self.mensaje_ui = None
        except Exception as e:
            # El audio falló por completo: lo mostramos en la UI y seguimos.
            self.mensaje_ui = f"Audio fallo ({self.motor}) - bin operativo"
            print(f"⚠️ Falló la voz ({self.motor}): {e}. Mensaje: {texto}")
        finally:
            self._hablando.clear()

    def _decir_piper(self, texto):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav = tmp.name
        try:
            subprocess.run(
                [self._piper_bin, "-m", self._piper_voice, "-f", wav],
                input=texto.encode("utf-8"),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(self._reproductor + [wav], check=True)
        finally:
            try:
                os.remove(wav)
            except OSError:
                pass

    def _decir_pyttsx3(self, texto):
        import pyttsx3

        # Un motor nuevo por frase es lo más estable al hablar desde un hilo.
        engine = pyttsx3.init()
        for voz in engine.getProperty("voices"):
            datos = f"{getattr(voz, 'id', '')} {getattr(voz, 'name', '')}".lower()
            if "spanish" in datos or "espanol" in datos or "es" in datos.split():
                engine.setProperty("voice", voz.id)
                break
        engine.say(texto)
        engine.runAndWait()
        engine.stop()
