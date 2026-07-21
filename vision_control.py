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
MOSTRAR_PREVIA = True         # False si el bin corre sin pantalla (modo headless)
VENTANA_DEBUG = "Smart Bin - depuracion en vivo"
TIEMPO_RESULTADO_MS = 900     # Tiempo para inspeccionar la decisión final en pantalla
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
        "probabilidades": probs,
    }


def normalizar_clase(clase):
    return clase.strip().lower()


def dibujar_debug(frame, estado, muestras=None, muestra_actual=None, comando=None):
    """Dibuja en la vista de cámara cada estado de la decisión del bin."""
    vista = frame.copy()
    alto, ancho = vista.shape[:2]
    color_estado = (0, 200, 0) if comando in (b"G", b"P") else (0, 190, 255)

    # Zona que conviene usar para colocar la botella durante las pruebas.
    lado = min(alto, ancho)
    x0, y0 = (ancho - lado) // 2, (alto - lado) // 2
    cv2.rectangle(vista, (x0, y0), (x0 + lado, y0 + lado), (120, 120, 120), 1)

    # Panel oscuro para que el texto sea visible con cualquier fondo.
    panel = vista.copy()
    cv2.rectangle(panel, (0, 0), (ancho, min(alto, 225)), (0, 0, 0), -1)
    vista = cv2.addWeighted(panel, 0.65, vista, 0.35, 0)

    cv2.putText(vista, estado, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color_estado, 2)
    cv2.putText(
        vista,
        f"Umbral: {UMBRAL_CONFIANZA:.0%} | q = salir",
        (18, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
    )

    if muestra_actual is not None:
        clase = muestra_actual["clase"]
        confianza = muestra_actual["confianza"]
        cv2.putText(
            vista,
            f"Muestra actual: {clase} ({confianza:.1%})",
            (18, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
        )
        for indice, clase in enumerate(CLASES):
            probabilidad = float(muestra_actual["probabilidades"][indice])
            y = 108 + indice * 27
            ancho_barra = int(probabilidad * 210)
            cv2.putText(
                vista,
                f"{clase:<7} {probabilidad:.1%}",
                (18, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.rectangle(vista, (155, y), (155 + ancho_barra, y + 16), (200, 200, 0), -1)

    if muestras:
        resumen = " | ".join(
            f"{indice + 1}:{muestra['clase']} {muestra['confianza']:.0%}"
            for indice, muestra in enumerate(muestras)
        )
        cv2.putText(
            vista,
            f"Muestras: {resumen}",
            (18, min(alto - 18, 213)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
    return vista


def mostrar_debug(frame, estado, muestras=None, muestra_actual=None, comando=None, espera_ms=1):
    """Muestra la interfaz y devuelve True cuando el usuario pide salir con q."""
    if not MOSTRAR_PREVIA:
        return False
    vista = dibujar_debug(frame, estado, muestras, muestra_actual, comando)
    cv2.imshow(VENTANA_DEBUG, vista)
    return cv2.waitKey(max(1, espera_ms)) & 0xFF == ord("q")


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

        if mostrar_debug(previa, "1/4 Esperando DETECTADO del sensor..."):
            break

        if arduino.in_waiting <= 0:
            continue

        # Usamos 'in' para tolerar los saltos de línea del Arduino.
        mensaje_arduino = arduino.readline().decode("utf-8", errors="replace").strip()
        if "DETECTADO" not in mensaje_arduino:
            continue

        print("\n🤖 [HARDWARE] Objeto detectado. Esperando estabilidad...")
        if MOSTRAR_PREVIA:
            if mostrar_debug(
                previa,
                "2/4 Objeto detectado: estabilizando...",
                espera_ms=int(ESPERA_ESTABILIZACION * 1000),
            ):
                break
        else:
            time.sleep(ESPERA_ESTABILIZACION)

        muestras = []
        salir_durante_muestras = False
        for numero in range(NUMERO_MUESTRAS):
            ret, frame = cap.read()
            muestra_actual = None
            if not ret:
                print(f"⚠️ No se pudo capturar la muestra {numero + 1}.")
                frame = previa
            else:
                try:
                    muestra_actual = clasificar_fotograma(frame, session, IN_NAME)
                    muestras.append(muestra_actual)
                    print(
                        f"  Muestra {numero + 1}/{NUMERO_MUESTRAS}: "
                        f"{muestra_actual['clase']} ({muestra_actual['confianza']:.1%})"
                    )
                except Exception as e:
                    print(f"⚠️ Error procesando la muestra {numero + 1}: {e}")

            estado = f"3/4 Analizando muestra {numero + 1}/{NUMERO_MUESTRAS}"
            if MOSTRAR_PREVIA:
                if mostrar_debug(
                    frame,
                    estado,
                    muestras,
                    muestra_actual,
                    espera_ms=int(INTERVALO_MUESTRAS * 1000),
                ):
                    salir_durante_muestras = True
                    break
            elif numero < NUMERO_MUESTRAS - 1:
                time.sleep(INTERVALO_MUESTRAS)

        if salir_durante_muestras:
            break

        mejor_muestra, comando = decidir_muestras(muestras)
        if mejor_muestra is None:
            print("❓ Descarte: ninguna muestra fue una clasificación válida de vidrio/plástico.")
            estado_final = "4/4 Resultado: RECHAZADO -> comando O"
            frame_final = muestras[-1]["frame"] if muestras else previa
            muestra_final = muestras[-1] if muestras else None
        else:
            cv2.imwrite("ultima_foto.jpg", mejor_muestra["frame"])
            etiqueta = "VIDRIO" if comando == b"G" else "PLÁSTICO"
            print(
                f"🎯 IA Decide: {etiqueta} "
                f"(mejor muestra: {mejor_muestra['confianza']:.1%}) -> "
                f"Enviando '{comando.decode()}'"
            )
            estado_final = f"4/4 Resultado: {etiqueta} -> comando {comando.decode()}"
            frame_final = mejor_muestra["frame"]
            muestra_final = mejor_muestra

        arduino.write(comando)
        arduino.flush()
        if MOSTRAR_PREVIA and mostrar_debug(
            frame_final,
            estado_final,
            muestras,
            muestra_final,
            comando,
            espera_ms=TIEMPO_RESULTADO_MS,
        ):
            break
        print("\nEsperando que el sensor ultrasónico detecte una botella...\n")
finally:
    cap.release()
    if MOSTRAR_PREVIA:
        cv2.destroyAllWindows()
    arduino.close()
