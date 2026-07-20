import serial
import time
import cv2
import numpy as np
import onnxruntime as ort
import json

# ================= CONFIGURACIÓN =================
PUERTO = '/dev/ttyUSB0'  # Ajusta si es necesario
BAUDIOS = 9600
RUTA_MODELO = 'bottle_classifier.onnx'
CONFIG_PATH = 'model_config.json'
# =================================================

# ----------------- 1. CARGAR CONFIGURACIÓN JSON -----------------
try:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    CLASES = cfg["classes"]
    SIZE = cfg["input_size"]
    MEAN = np.array(cfg["mean"], dtype=np.float32)
    STD = np.array(cfg["std"], dtype=np.float32)
    print(f"✅ Configuración cargada: {CLASES} | Tamaño: {SIZE}x{SIZE}")
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
    print(f"✅ Modelo cargado y listo.")
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    exit()

print("\n🚀 --- SISTEMA DE CLASIFICACIÓN ACTIVO --- 🚀")
print("Esperando que el sensor ultrasónico detecte una botella...\n")

# ----------------- 4. BUCLE PRINCIPAL -----------------
while True:
    if arduino.in_waiting > 0:
        # Usamos 'in' en lugar de '==' para ser más tolerantes con los saltos de línea
        mensaje_arduino = arduino.readline().decode('utf-8').strip()
        
        if "DETECTADO" in mensaje_arduino:
            print("\n🤖 [HARDWARE] Objeto detectado. Iniciando cámara...")
            
            # --- CAPTURA DE FOTO ---
            cap = cv2.VideoCapture(0)
            for _ in range(10): # Calentamiento de cámara
                cap.read()
                time.sleep(0.05)
                
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("⚠️ Error con la cámara. Abortando.")
                arduino.write(b'O')
                arduino.flush()
                continue
                
            print("📸 Foto capturada. Analizando...")
            cv2.imwrite("ultima_foto.jpg", frame) 
            
            # --- INFERENCIA CON ONNX ---
            try:
                input_data = preprocesar_imagen(frame)
                
                logits = session.run(None, {IN_NAME: input_data})[0][0]
                probs = softmax(logits) 
                
                prediccion_idx = int(probs.argmax())
                confianza = float(probs[prediccion_idx])
                clase_detectada = CLASES[prediccion_idx]
                
                # --- TOMA DE DECISIÓN (SIN SLEEPS QUE MATEN EL CICLO) ---
                if confianza < 0.75 or clase_detectada == "other": 
                    print(f"❓ Descarte -> Clase: {clase_detectada} | Seguridad: {confianza*100:.1f}%")
                    arduino.write(b'O')
                    arduino.flush() # Obliga a enviar instantáneamente
                    
                elif clase_detectada == "glass" or clase_detectada == "Vidrio":
                    print(f"🎯 IA Decide: VIDRIO (Seguridad: {confianza*100:.1f}%) -> Enviando 'G'")
                    arduino.write(b'G')
                    arduino.flush()
                    
                elif clase_detectada == "plastic" or clase_detectada == "Plastico":
                    print(f"🎯 IA Decide: PLÁSTICO (Seguridad: {confianza*100:.1f}%) -> Enviando 'P'")
                    arduino.write(b'P')
                    arduino.flush()
                    
                print("\nEsperando que el sensor ultrasónico detecte una botella...\n")
                    
            except Exception as e:
                print(f"⚠️ Error procesando la imagen: {e}")
                arduino.write(b'O')
                arduino.flush()