#include <Servo.h>

Servo rampaPrincipal;
const int pinServo = 9;

const int trigPin = 10;
const int echoPin = 11;
const int distanciaUmbral = 15; // Centímetros

enum EstadoSistema {
  ESPERANDO_BOTELLA,
  ESPERANDO_COMANDO_PYTHON
};

EstadoSistema estadoActual = ESPERANDO_BOTELLA;

// --- TIEMPOS INDEPENDIENTES PARA CALIBRACIÓN FINA ---

// Lado PLÁSTICO (Estos ya te funcionan bien, ¡no los toques!)
int tiempoCaidaP = 300;  
int tiempoSubidaP = 380; 

// Lado VIDRIO (Aquí hacemos la magia para compensar el motor)
// Como no bajaba mucho, le subimos el tiempo. Como regresaba mucho, le bajamos el tiempo.
int tiempoCaidaG = 380; // Antes era 300
int tiempoSubidaG = 320; // Antes era 380
// ----------------------------------------------------

void setup() {
  rampaPrincipal.attach(pinServo);
  rampaPrincipal.write(90); // FRENO TOTAL

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  Serial.begin(9600);
  Serial.println("Sistema listo. Esperando botella...");
}

void loop() {
  
  if (estadoActual == ESPERANDO_BOTELLA) {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    
    // --- EL SALVAVIDAS ---
    // Limite de 30ms para que el Arduino no se congele si falla el eco
    long duracion = pulseIn(echoPin, HIGH, 30000); 
    long distancia = duracion * 0.034 / 2;

    if (distancia > 0 && distancia <= distanciaUmbral) {
      Serial.println("DETECTADO"); 
      estadoActual = ESPERANDO_COMANDO_PYTHON; 
    }
    delay(100); 
  }
  
  else if (estadoActual == ESPERANDO_COMANDO_PYTHON) {
    if (Serial.available() > 0) {
      char comando = Serial.read();
      comando = toupper(comando);
      
      if (comando == 'G') { // --- VIDRIO ---
        Serial.println("[OK] Inclinando rampa para Vidrio...");
        
        rampaPrincipal.write(110); 
        delay(tiempoCaidaG);        
        rampaPrincipal.write(90);  
        
        delay(1500); 
        
        Serial.println("[OK] Nivelando rampa...");
        rampaPrincipal.write(70);  
        delay(tiempoSubidaG);       
        rampaPrincipal.write(90);  
        
        delay(500);  
        estadoActual = ESPERANDO_BOTELLA; 
        Serial.println("Listo para la siguiente botella...");
        
      } else if (comando == 'P') { // --- PLÁSTICO ---
        Serial.println("[OK] Inclinando rampa para Plastico...");
        
        rampaPrincipal.write(70);  
        delay(tiempoCaidaP);        
        rampaPrincipal.write(90);  
        
        delay(1500); 
        
        Serial.println("[OK] Nivelando rampa...");
        rampaPrincipal.write(110); 
        delay(tiempoSubidaP);       
        rampaPrincipal.write(90);  
        
        delay(500); 
        estadoActual = ESPERANDO_BOTELLA; 
        Serial.println("Listo para la siguiente botella...");
        
      } else if (comando == 'O') { // --- DESCARTE ---
        Serial.println("[OK] Descarte (O). Por favor retire el objeto...");
        
        // --- SISTEMA ANTI-SPAM ---
        // Se queda atrapado en este bucle HASTA que quites la mano/objeto
        long distanciaSeguridad = 0;
        do {
          digitalWrite(trigPin, LOW);
          delayMicroseconds(2);
          digitalWrite(trigPin, HIGH);
          delayMicroseconds(10);
          digitalWrite(trigPin, LOW);
          
          long dur = pulseIn(echoPin, HIGH, 30000);
          distanciaSeguridad = dur * 0.034 / 2;
          
          delay(200); // Lee el sensor despacito mientras esperas
          
        } while (distanciaSeguridad > 0 && distanciaSeguridad <= distanciaUmbral);
        
        // Si el código llega aquí, es porque la distancia ya es mayor a 15cm (el camino está libre)
        delay(1000); // Un segundito de margen extra por si acaso
        
        estadoActual = ESPERANDO_BOTELLA; 
        Serial.println("Listo para la siguiente botella...");
      }

      // --- LIMPIAPARABRISAS DEL BUFFER ---
      // Bota a la basura cualquier letra invisible que haya sobrado para que no trabe el ciclo
      while(Serial.available() > 0) { 
        Serial.read(); 
      }
    }
  }
}