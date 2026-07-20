import json
import time

import cv2
import numpy as np
import onnxruntime as ort
import serial

# ================= CONFIGURACIÓN =================
PUERTO = '/dev/ttyUSB0'  # Ajusta si es necesario
BAUDIOS = 9600
RUTA_MODELO = 'bottle_classifier.onnx'
CONFIG_PATH = 'model_config.json'

INDICE_CAMARA = 0
NUMERO_MUESTRAS = 5           # Fotogramas analizados por cada objeto detectado
ESPERA_ESTABILIZACION = 0.20  # Da tiempo a que el objeto deje de moverse
INTERVALO_MUESTRAS = 0.08
MOSTRAR_PREVIA = False        # Actívalo solo si el equipo tiene pantalla
# =================================================


# ----------------- 1. CARGAR CONFIGURACIÓN JSON -----------------
try:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    CLASES = cfg["classes"]
    SIZE = cfg["input_size"]
    MEAN = np.array(cfg["mean"], dtype=np.float32)
    STD = np.array(cfg["std"], dtype=np.float32)
    UMBRAL_CONFIANZA = float(cfg["confidence_threshold"])
    print(
        f"✅ Configuración cargada: {CLASES} | Tamaño: {SIZE}x{SIZE} | "
        f"Umbral: {UMBRAL_CONFIANZA:.0%}"
    )
except Exception as e:
    print(f"❌ Error al cargar {CONFIG_PATH}: {e}")
    exit()


def preprocesar_imagen(frame_bgr):
    img = cv2.resize(frame_bgr, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))
    return img[np.newaxis, :].astype(np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def clasificar_fotograma(frame_bgr, session, input_name):
    """Devuelve la clase y confianza de un único fotograma de la cámara."""
    input_data = preprocesar_imagen(frame_bgr)
    logits = session.run(None, {input_name: input_data})[0][0]
    probs = softmax(logits)
    indice = int(probs.argmax())

    return {
        "frame": frame_bgr,
        "clase": CLASES[indice],
        "confianza": float(probs[indice]),
    }


def normalizar_clase(clase):
    return clase.strip().lower()


def decidir_muestras(muestras):
    """Vota entre las muestras válidas y devuelve la mejor de la clase ganadora."""
    comandos = {"glass": b"G", "vidrio": b"G", "plastic": b"P", "plastico": b"P"}
    candidatas = [
        muestra
        for muestra in muestras
        if muestra["confianza"] >= UMBRAL_CONFIANZA
        and normalizar_clase(muestra["clase"]) in comandos
    ]

    if not candidatas:
        return None, b"O"

    conteos = {}
    for muestra in candidatas:
        clase = normalizar_clase(muestra["clase"])
        conteos[clase] = conteos.get(clase, 0) + 1

    # Si hay empate, gana la clase con mayor suma de confianza.
    clase_ganadora = max(
        conteos,
        key=lambda clase: (
            conteos[clase],
            sum(m["confianza"] for m in candidatas if normalizar_clase(m["clase"]) == clase),
        ),
    )
    mejor_muestra = max(
        (m for m in candidatas if normalizar_clase(m["clase"]) == clase_ganadora),
        key=lambda muestra: muestra["confianza"],
    )
    return mejor_muestra, comandos[clase_ganadora]


# ----------------- 2. INICIAR HARDWARE -----------------
try:
    print(f"🔌 Conectando al Nano en {PUERTO}...")
    arduino = serial.Serial(PUERTO, BAUDIOS, timeout=1)
    time.sleep(2)
    print("✅ Conexión serial establecida.")
except Exception as e:
    print(f"❌ Error de hardware: {e}")
    exit()

# ----------------- 3. CARGAR IA -----------------
try:
    print("🧠 Cargando modelo ONNX...")
    session = ort.InferenceSession(RUTA_MODELO, providers=["CPUExecutionProvider"])
    IN_NAME = session.get_inputs()[0].name
    print("✅ Modelo cargado y listo.")
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    arduino.close()
    exit()

# ----------------- 4. INICIAR CÁMARA CONTINUA -----------------
cap = cv2.VideoCapture(INDICE_CAMARA)
if not cap.isOpened():
    print("❌ Error al abrir la cámara.")
    arduino.close()
    exit()

print("📷 Cámara lista y capturando continuamente.")
print("\n🚀 --- SISTEMA DE CLASIFICACIÓN ACTIVO --- 🚀")
print("Esperando que el sensor ultrasónico detecte una botella...\n")

# ----------------- 5. BUCLE PRINCIPAL -----------------
try:
    while True:
        # Mantener la cámara leyendo evita abrirla, enfocarla y cerrarla por cada objeto.
        ret, previa = cap.read()
        if not ret:
            print("⚠️ No se pudo leer la cámara.")
            time.sleep(0.1)
            continue

        if MOSTRAR_PREVIA:
            cv2.imshow("Vista previa - presiona q para salir", previa)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if arduino.in_waiting <= 0:
            continue

        # Usamos 'in' para tolerar los saltos de línea del Arduino.
        mensaje_arduino = arduino.readline().decode("utf-8", errors="replace").strip()
        if "DETECTADO" not in mensaje_arduino:
            continue

        print("\n🤖 [HARDWARE] Objeto detectado. Esperando estabilidad...")
        time.sleep(ESPERA_ESTABILIZACION)

        muestras = []
        for numero in range(NUMERO_MUESTRAS):
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ No se pudo capturar la muestra {numero + 1}.")
            else:
                try:
                    muestra = clasificar_fotograma(frame, session, IN_NAME)
                    muestras.append(muestra)
                    print(
                        f"  Muestra {numero + 1}/{NUMERO_MUESTRAS}: "
                        f"{muestra['clase']} ({muestra['confianza']:.1%})"
                    )
                except Exception as e:
                    print(f"⚠️ Error procesando la muestra {numero + 1}: {e}")

            if numero < NUMERO_MUESTRAS - 1:
                time.sleep(INTERVALO_MUESTRAS)

        mejor_muestra, comando = decidir_muestras(muestras)
        if mejor_muestra is None:
            print("❓ Descarte: ninguna muestra válida superó el umbral o fue vidrio/plástico.")
        else:
            cv2.imwrite("ultima_foto.jpg", mejor_muestra["frame"])
            etiqueta = "VIDRIO" if comando == b"G" else "PLÁSTICO"
            print(
                f"🎯 IA Decide: {etiqueta} "
                f"(mejor muestra: {mejor_muestra['confianza']:.1%}) -> "
                f"Enviando '{comando.decode()}'"
            )

        arduino.write(comando)
        arduino.flush()
        print("\nEsperando que el sensor ultrasónico detecte una botella...\n")
finally:
    cap.release()
    if MOSTRAR_PREVIA:
        cv2.destroyAllWindows()
    arduino.close()
