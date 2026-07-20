#!/usr/bin/env python3
"""
test_cdo testdo testamara.py - Prueba en vivo de la camara + modelo del Smart Recycling Bin.

Objetivo: verificar la CALIDAD de la camara y si el modelo distingue
vidrio / plastico / other en tiempo real, ANTES de construir el rig.

IMPORTANTE:do testdo test esto corre LOCAL en tu laptop, NO en Google Colab.
Colab no puede acceder a tu webcam USB del bin.
do test
Requisitos (en tu laptop):
    pip install onnxruntime opencv-python numpy

Uso:
    python test_camara.py
    Coloca junto a este script:  bottle_classifier.onnx  y  model_config.json
    Teclas:  q = salir  |  s = guardar frame  |  c = cambiar de camara
"""

import json
import cv2
import numpy as np
import onnxruntime as ort

# --------------------------------------------------------------------------
# Config editable
# --------------------------------------------------------------------------
MODEL_PATH  = "bottle_classifier.onnx"
CONFIG_PATH = "model_config.json"
CAM_INDEX   = 2          # 0 = webcam integrada; prueba 1, 2... para la USB del bin

# --------------------------------------------------------------------------
# Cargar config y modelo
# --------------------------------------------------------------------------
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

CLASSES = cfg["classes"]                   # NUNCA hardcodear: se lee del config
THRESH  = cfg["confidence_threshold"]      # 0.70
SIZE    = cfg["input_size"]                # 224
MEAN    = np.array(cfg["mean"], dtype=np.float32)
STD     = np.array(cfg["std"],  dtype=np.float32)

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
IN_NAME = session.get_inputs()[0].name     # 'input'

print(f"Clases (en el orden del modelo): {CLASSES}")
print(f"Umbral de confianza: {THRESH}")


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def preprocess(frame_bgr):
    """Replica val_transforms del notebook: Resize((224,224)) + ToTensor + Normalize."""
    img = cv2.resize(frame_bgr, (SIZE, SIZE))       # redimension al cuadrado (igual que entrenamiento)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # BGR -> RGB (OBLIGATORIO)
    img = img.astype(np.float32) / 255.0            # escala a [0,1]
    img = (img - MEAN) / STD                        # Normalize ImageNet
    img = np.transpose(img, (2, 0, 1))              # HWC -> CHW
    return img[np.newaxis, :].astype(np.float32)    # batch


def classify(frame_bgr):
    x = preprocess(frame_bgr)
    logits = session.run(None, {IN_NAME: x})[0][0]  # logits crudos
    probs = softmax(logits)
    idx = int(probs.argmax())
    conf = float(probs[idx])
    label = CLASSES[idx]
    routed = label if (conf >= THRESH and label != "other") else "RECHAZADO"
    return label, conf, probs, routed


def open_cam(idx):
    cap = cv2.VideoCapture(idx)
    return cap if cap.isOpened() else None


def draw(frame, label, conf, probs, routed):
    ok_route = routed in ("glass", "plastic")
    color = (0, 200, 0) if ok_route else (0, 0, 255)
    h, w = frame.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (120, 120, 120), 1)
    cv2.putText(frame, f"{routed}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f"argmax: {label}  conf: {conf:.2f}  (umbral {THRESH})", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, cls in enumerate(CLASSES):
        bar = int(probs[i] * 200)
        y = 110 + i * 26
        cv2.putText(frame, f"{cls:<7} {probs[i]:.2f}", (20, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (150, y), (150 + bar, y + 16), (200, 200, 0), -1)
    return frame


def main():
    global CAM_INDEX
    cap = open_cam(CAM_INDEX)
    if cap is None:
        raise SystemExit(f"No pude abrir la camara {CAM_INDEX}. Prueba CAM_INDEX = 1, 2...")
    print("Camara abierta. q=salir  s=guardar  c=cambiar camara")
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No llega frame de la camara.")
            break
        label, conf, probs, routed = classify(frame)
        frame = draw(frame, label, conf, probs, routed)
        cv2.imshow("Smart Bin - test camara (q=salir)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"captura_{saved:02d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Guardado: {fname}")
            saved += 1
        elif key == ord("c"):
            CAM_INDEX += 1
            cap.release()
            nc = open_cam(CAM_INDEX)
            if nc is None:
                print(f"No hay camara {CAM_INDEX}, vuelvo a 0.")
                CAM_INDEX = 0
                nc = open_cam(0)
            cap = nc
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
