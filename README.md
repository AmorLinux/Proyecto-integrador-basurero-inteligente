# Basurero Inteligente (Smart Recycling Bin)

Sistema que clasifica objetos en **vidrio** / **plástico** / **otro** con una
cámara y un modelo de IA (ONNX), y le indica a un Arduino Nano cómo separarlos.
Cuando el objeto **no** es vidrio ni plástico, la laptop lo **avisa por voz**.

---

## Cómo funciona

1. La cámara muestra vídeo en vivo y clasifica cada fotograma (vista de depuración).
2. El sensor ultrasónico del Arduino manda `DETECTADO` cuando aparece un objeto.
3. Tras un tiempo de **asentamiento** (para que la botella se coloque bien), se
   toman varias muestras y se **vota** la clase.
4. Se envía un comando al Arduino: `G` = vidrio, `P` = plástico, `O` = rechazado.
5. Si el resultado es rechazado (`O`), la laptop lo **dice en voz alta**.

---

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `vision_control.py` | Programa principal: cámara + IA + serial + voz. |
| `voz.py` | Módulo de voz offline (Piper o pyttsx3), no bloqueante. |
| `test_camara.py` | Prueba sólo la cámara + modelo (sin Arduino). Útil para calibrar. |
| `prueba_motor.py` | Prueba de los motores/servos vía Arduino. |
| `bottle_classifier.onnx` | Modelo de clasificación entrenado. |
| `model_config.json` | Clases, umbral y normalización (NO se hardcodean en el código). |
| `ML.ipynb` | Notebook de entrenamiento del modelo. |

---

## Instalación

### 1. Dependencias de Python

```bash
pip install onnxruntime opencv-python numpy pyserial
```

> Usa `opencv-python`, **no** `opencv-python-headless`: la versión headless no
> puede abrir la ventana de vídeo.

### 2. Voz (elige UNA opción)

**Opción fácil — voz robótica, se instala al instante:**

```bash
pip install pyttsx3
sudo apt install espeak-ng          # Linux / WSL
```

**Opción recomendada — voz natural en español (Piper):**

```bash
pip install piper-tts
sudo apt install pulseaudio-utils    # aporta 'paplay' para reproducir el audio
# Descarga una voz en español (.onnx + .onnx.json), p. ej. es_MX-* o es_ES-davefx-medium
export PIPER_VOICE=/ruta/a/es_MX-voz.onnx
```

Al arrancar verás qué motor se activó:
`🔊 Voz activada (motor: piper)` o `pyttsx3`, o bien `🔇 Sin motor de voz…`.
Si no hay motor o el audio falla, el bin **sigue funcionando** y muestra un
aviso rojo en la ventana; nunca se detiene por la voz.

---

## Configuración

Ajusta las constantes al inicio de `vision_control.py`:

| Constante | Default | Qué controla |
|---|---|---|
| `PUERTO` | `/dev/ttyUSB0` | Puerto serial del Arduino. |
| `INDICE_CAMARA` | `2` | Cámara a usar (0 = integrada; prueba 1, 2…). |
| `ESPERA_ESTABILIZACION` | `2` | Segundos que espera tras `DETECTADO` para que el objeto se asiente antes de clasificar. |
| `NUMERO_MUESTRAS` | `5` | Fotogramas que analiza y vota por objeto. |
| `MOSTRAR_PREVIA` | `True` | `False` para correr sin pantalla (headless). |
| `VOZ_ACTIVA` | `True` | Activa/desactiva la voz. |
| `MENSAJE_RECHAZO` | *"Objeto no reconocido…"* | Frase que dice al rechazar. |
| `COOLDOWN_VOZ` | `8.0` | Segundos mínimos entre un aviso de voz y el siguiente. |

El umbral de confianza y las clases se leen de `model_config.json`.

---

## Uso

```bash
python3 vision_control.py
```

Con el Arduino conectado, coloca un objeto frente al sensor y observa la ventana
de depuración en vivo.

**Teclas / controles:**

- `q` — salir.

**Para calibrar sólo la cámara y el modelo (sin Arduino):**

```bash
python3 test_camara.py
```
Teclas: `q` = salir · `s` = guardar frame · `c` = cambiar de cámara.

---

## Solución de problemas

| Síntoma | Causa probable / solución |
|---|---|
| No abre la ventana de vídeo | Tienes `opencv-python-headless`; instala `opencv-python`. En WSL necesitas WSLg (o un servidor X). |
| Clasifica con el objeto aún en el aire | Sube `ESPERA_ESTABILIZACION`. |
| No abre la cámara | Cambia `INDICE_CAMARA` (0, 1, 2…). |
| No conecta al Arduino | Revisa `PUERTO` (`/dev/ttyUSB0`, `/dev/ttyACM0`, `COM3`…) y permisos. |
| Aviso rojo "Voz no disponible" | No hay motor TTS: instala `pyttsx3` o configura Piper. |
| Aviso rojo "Audio fallo" | Hay motor pero no suena: en WSL el audio va por WSLg; verifica `paplay`/altavoces. |
