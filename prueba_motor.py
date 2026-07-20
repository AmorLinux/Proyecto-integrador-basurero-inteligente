import serial
import time

# El puerto se mantiene igual para tu entorno Linux
PUERTO = '/dev/ttyUSB0'  
BAUDIOS = 9600

# 1. Iniciar conexión
try:
    print(f"🔌 Conectando al Arduino en {PUERTO}...")
    arduino = serial.Serial(PUERTO, BAUDIOS, timeout=1)
    time.sleep(2)  # Pausa para que el Arduino se reinicie
    print("✅ ¡Conexión establecida con el Hardware!")
except Exception as e:
    print(f"❌ Error de conexión: {e}")
    exit()

print("\n⚙️ --- PRUEBA MANUAL DE SERVOMOTOR DE ALTA POTENCIA --- ⚙️")
print("1. Pasa tu mano por el sensor ultrasónico para 'despertar' al Arduino.")
print("2. Escribe G, P u O cuando el sistema te lo pida.\n")

# 2. Bucle de prueba
while True:
    try:
        if arduino.in_waiting > 0:
            mensaje = arduino.readline().decode('utf-8').strip()
            
            # Imprimimos lo que hace el Arduino por detrás
            if mensaje and mensaje != "DETECTADO":
                print(f"   [ARDUINO]: {mensaje}")
            
            # Si el sensor se activa, pedimos el comando manual
            if mensaje == "DETECTADO":
                print("\n🤖 ¡Sensor activado! El Arduino está esperando tu orden.")
                comando = input("👉 Escribe G (Vidrio), P (Plástico) u O (Cancelar): ").strip().upper()
                
                if comando in ['G', 'P', 'O']:
                    arduino.write(comando.encode('utf-8'))
                    print(f"⚡ Comando '{comando}' enviado. ¡Observa la rampa!")
                else:
                    print("🚫 Comando no válido. Enviando descarte (O)...")
                    arduino.write(b'O')
                
                # Damos tiempo al motor para moverse y limpiamos los fantasmas de la memoria
                time.sleep(2)
                arduino.reset_input_buffer()
                print("\nEsperando siguiente activación del sensor...\n")
                
    except KeyboardInterrupt:
        print("\n\nSaliendo del programa de prueba...")
        arduino.close()
        break
