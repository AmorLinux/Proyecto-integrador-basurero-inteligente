import json
import time

import cv2
import numpy as np
import onnxruntime as ort
import serial

from voz import Locutor

# ================= CONFIGURACIÓN =================
PUERTO = '/dev/ttyUSB0'  # Ajusta si es necesario
BAUDIOS = 9600
RUTA_MODELO = 'bottle_classifier.onnx'
CONFIG_PATH = 'model_config.json'

INDICE_CAMARA = 2
NUMERO_MUESTRAS = 5           # Fotogramas analizados por cada objeto detectado
# Tiempo que espera DESPUÉS del DETECTADO del sensor antes de clasificar, para dar
# tiempo a que la botella termine de caer/asentarse en su sitio. El sensor dispara
# apenas ve algo, así que sin esta espera clasificaría con el objeto aún en el aire.
# Súbelo si la botella tarda más en quedar bien colocada.
ESPERA_ESTABILIZACION = 2
INTERVALO_MUESTRAS = 0.08
MOSTRAR_PREVIA = True         # False si el bin corre sin pantalla (modo headless)
VENTANA_DEBUG = "Smart Bin - depuracion en vivo"
TIEMPO_RESULTADO_MS = 900     # Tiempo para inspeccionar la decisión final en pantalla

# --- Voz cuando el objeto NO es vidrio ni plástico (comando O / RECHAZADO) ---
VOZ_ACTIVA = True
MENSAJE_RECHAZO = "Objeto no reconocido. No es vidrio ni plástico. Por favor, retíralo."
COOLDOWN_VOZ = 8.0            # Segundos mínimos entre un aviso de voz y el siguiente
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


def leer_y_clasificar(cap, session, input_name):
    """Lee un fotograma y lo clasifica. Devuelve (frame, muestra) o (None, None)."""
    ret, frame = cap.read()
    if not ret:
        return None, None
    try:
        muestra = clasificar_fotograma(frame, session, input_name)
    except Exception as e:
        print(f"⚠️ Error clasificando fotograma: {e}")
        muestra = {
            "frame": frame,
            "clase": "?",
            "confianza": 0.0,
            "probabilidades": np.zeros(len(CLASES), dtype=np.float32),
        }
    return frame, muestra


def normalizar_clase(clase):
    return clase.strip().lower()


def dibujar_debug(frame, estado, muestras=None, muestra_actual=None, comando=None, aviso=None):
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

    # Banner inferior de aviso (p. ej. audio caído). El bin sigue operando;
    # esto es sólo informativo para depurar sin mirar el terminal.
    if aviso:
        y1 = max(0, alto - 30)
        sub = vista[y1:alto]
        rojo = np.full_like(sub, (0, 0, 150))
        vista[y1:alto] = cv2.addWeighted(rojo, 0.55, sub, 0.45, 0)
        cv2.putText(vista, f"! {aviso}", (18, alto - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vista


_previa_deshabilitada = False


def mostrar_debug(frame, estado, muestras=None, muestra_actual=None, comando=None, espera_ms=1):
    """Muestra la interfaz y devuelve True cuando el usuario pide salir con q."""
    global _previa_deshabilitada
    if not MOSTRAR_PREVIA or _previa_deshabilitada:
        return False
    # Si la voz falló o no hay motor, el Locutor lo expone aquí para pintarlo.
    aviso = getattr(locutor, "mensaje_ui", None) if "locutor" in globals() and locutor else None
    try:
        vista = dibujar_debug(frame, estado, muestras, muestra_actual, comando, aviso)
        cv2.imshow(VENTANA_DEBUG, vista)
        return cv2.waitKey(max(1, espera_ms)) & 0xFF == ord("q")
    except cv2.error as e:
        # Sin esto la ventana falla en silencio (p. ej. opencv-python-headless
        # o sin servidor X) y sólo se ven los logs del terminal.
        print(
            f"⚠️ No se pudo mostrar la ventana '{VENTANA_DEBUG}': {e}\n"
            "   Revisa que tengas pantalla/servidor X y 'opencv-python' "
            "(no '-headless'). Sigo en modo sólo-terminal."
        )
        _previa_deshabilitada = True
        return False


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

# ----------------- 3b. INICIAR VOZ -----------------
# Si la voz no puede iniciarse no pasa nada: el Locutor degrada a sólo texto.
locutor = Locutor(cooldown=COOLDOWN_VOZ) if VOZ_ACTIVA else None

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
# La cámara se lee y clasifica en CADA vuelta del bucle (igual que test_camara.py),
# así la ventana siempre muestra vídeo en vivo y nunca se congela mientras esperamos
# al sensor. El sensor sólo dispara la ráfaga de muestras y el comando al Arduino.
try:
    while True:
        frame, muestra_live = leer_y_clasificar(cap, session, IN_NAME)
        if frame is None:
            print("⚠️ No se pudo leer la cámara.")
            time.sleep(0.1)
            continue

        # ¿El sensor reportó un objeto? Usamos 'in' para tolerar saltos de línea.
        detectado = False
        if arduino.in_waiting > 0:
            mensaje_arduino = arduino.readline().decode("utf-8", errors="replace").strip()
            detectado = "DETECTADO" in mensaje_arduino

        if not detectado:
            # Vista en vivo continua mientras esperamos al sensor.
            if mostrar_debug(frame, "En vivo | Esperando DETECTADO del sensor...",
                             muestra_actual=muestra_live):
                break
            continue

        print(
            f"\n🤖 [HARDWARE] Objeto detectado. Esperando "
            f"{ESPERA_ESTABILIZACION:.1f}s a que se coloque..."
        )

        # Espera de asentamiento SIN congelar: seguimos leyendo y mostrando frames
        # en vivo, con una cuenta regresiva para ver cuánto falta para clasificar.
        salir = False
        t_fin = time.time() + ESPERA_ESTABILIZACION
        while time.time() < t_fin:
            frame, muestra_live = leer_y_clasificar(cap, session, IN_NAME)
            if frame is None:
                continue
            restante = max(0.0, t_fin - time.time())
            if mostrar_debug(frame, f"Coloca el objeto... clasifico en {restante:.1f}s",
                             muestra_actual=muestra_live):
                salir = True
                break
        if salir:
            break

        # Ráfaga de muestras para votar la clase.
        muestras = []
        for numero in range(NUMERO_MUESTRAS):
            frame, muestra_actual = leer_y_clasificar(cap, session, IN_NAME)
            if frame is None:
                print(f"⚠️ No se pudo capturar la muestra {numero + 1}.")
                continue
            if muestra_actual["clase"] != "?":
                muestras.append(muestra_actual)
                print(
                    f"  Muestra {numero + 1}/{NUMERO_MUESTRAS}: "
                    f"{muestra_actual['clase']} ({muestra_actual['confianza']:.1%})"
                )

            estado = f"Analizando muestra {numero + 1}/{NUMERO_MUESTRAS}"
            if mostrar_debug(
                frame,
                estado,
                muestras,
                muestra_actual,
                espera_ms=int(INTERVALO_MUESTRAS * 1000),
            ):
                salir = True
                break
        if salir:
            break

        mejor_muestra, comando = decidir_muestras(muestras)
        if mejor_muestra is None:
            print("❓ Descarte: ninguna muestra fue una clasificación válida de vidrio/plástico.")
            estado_final = "Resultado: RECHAZADO -> comando O"
            muestra_final = muestras[-1] if muestras else None
            # Aviso por voz (en un hilo aparte; el cooldown evita repetir muy seguido).
            if locutor is not None:
                locutor.decir(MENSAJE_RECHAZO)
        else:
            cv2.imwrite("ultima_foto.jpg", mejor_muestra["frame"])
            etiqueta = "VIDRIO" if comando == b"G" else "PLÁSTICO"
            print(
                f"🎯 IA Decide: {etiqueta} "
                f"(mejor muestra: {mejor_muestra['confianza']:.1%}) -> "
                f"Enviando '{comando.decode()}'"
            )
            estado_final = f"Resultado: {etiqueta} -> comando {comando.decode()}"
            muestra_final = mejor_muestra

        arduino.write(comando)
        arduino.flush()

        # Mostrar el resultado SIN congelar: seguimos leyendo frames en vivo mientras
        # se rotula la decisión final durante TIEMPO_RESULTADO_MS.
        t_fin = time.time() + TIEMPO_RESULTADO_MS / 1000
        while time.time() < t_fin:
            frame_live, _ = leer_y_clasificar(cap, session, IN_NAME)
            if frame_live is None:
                continue
            if mostrar_debug(frame_live, estado_final, muestras, muestra_final, comando):
                salir = True
                break
        if salir:
            break

        # Descartamos los DETECTADO que el sensor haya encolado durante todo el
        # ciclo, para no re-disparar de inmediato con lecturas viejas.
        arduino.reset_input_buffer()
        print("\nEsperando que el sensor ultrasónico detecte una botella...\n")
finally:
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
