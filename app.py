import requests
import pyttsx3
import speech_recognition as sr
import shutil
import os
import tempfile
import datetime
import whisper
import threading
import json

url = "http://localhost:8080/ask"
urlDoc = "http://localhost:8080/docs"
log_file = 'docs/log.txt'

file = tempfile.mkdtemp()
path = os.path.join(file, 'temp.wav')

engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty('rate', 145)
engine.setProperty('voice', voices[0].id)

clave = 'acho'

avisos = []

model = whisper.load_model("base")

def talk(text):
    engine.say(text)
    engine.runAndWait()

def esperando_clave():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Esperando palabra clave...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            with open(path, "wb") as f:
                f.write(audio.get_wav_data())

            result = model.transcribe(path, language="spanish")
            detected_text = result["text"].strip().lower()

            if clave in detected_text:
                print("Palabra clave detectada. Escuchando...")
                return True, detected_text[detected_text.index(clave) + len(clave):].strip()
            else:
                return False, ""
        except sr.WaitTimeoutError:
            print("No se detectó ninguna entrada de voz.")
        except Exception as e:
            print(f"Error durante la detección de la palabra clave: {e}")
    return False, ""

def conversacion_fluida():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Escuchando...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            with open(path, "wb") as f:
                f.write(audio.get_wav_data())

            result = model.transcribe(path, language="spanish")
            detected_text = result["text"].strip().lower()
            return detected_text
        except sr.WaitTimeoutError:
            print("No se detectó ninguna entrada de voz.")
        except Exception as e:
            print(f"Error durante la conversación fluida: {e}")
    return ""

def borrarBD():
    folder = "db"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    print("Base de datos borrada")

def cargarMedicamentos():
    try:
        with open('docs/medicamentos.txt', 'rb') as archivo:
            files = {"file": archivo}
            response = requests.post(urlDoc, files=files)

            if response.status_code == 200:
                print("Medicamentos cargados")
            else:
                print(f"Error al enviar archivo: Código de estado {response.status_code}")
    except Exception as e:
        print(f"Error al cargar medicamentos: {e}")

def horaActual():
    return datetime.datetime.now().strftime("%H:%M")

def registrar_evento(mensaje):
    try:
        with open(log_file, 'a') as log:
            fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            log.write(f"{fecha} - {mensaje}\n")
    except Exception as e:
        print(f"Error al escribir en el log: {e}")

def existe_aviso_para_medicamento(nombre_medicamento):
    for aviso in avisos:
        if aviso[0] == nombre_medicamento:
            return True
    return False

def borrar_avisos():
    now = datetime.datetime.now().strftime("%H:%M")
    
    for aviso in avisos:
        if (aviso[2] == -1 or aviso[2] == 3) and aviso[1].strftime("%H:%M") != now:
            avisos.remove(aviso)

def verificar_medicamentos():
    try:
        with open('docs/medicamentos.txt', 'r') as archivo:
            medicamentos = json.load(archivo)

        current_time = horaActual()

        for med in medicamentos:
            if current_time in med["hours"] and not existe_aviso_para_medicamento(med['name']):
                anunciar_aviso = f"Es hora de tomar su medicamento: {med['name']} ({med['alt_name']})."
                registrar_evento(anunciar_aviso)
                avisos.append((med['name'], datetime.datetime.now(), 0))
                print(anunciar_aviso)
                talk(anunciar_aviso)

    except Exception as e:
        print(f"Error en la verificación de medicamentos: {e}")


def verificar_tomas():
    borrar_avisos()

    now = datetime.datetime.now()
    
    for aviso in avisos:
        med_name, notif_time, veces = aviso
        if (now - notif_time).total_seconds() >= 15 and veces != -1 and veces != 3:
            print(f"¿Ha tomado su medicamento {med_name}?")
            talk(f"¿Ha tomado su medicamento {med_name}?")
            respuesta = conversacion_fluida()
            print("Usuario: " + respuesta)
            if "sí" in respuesta.lower() or "si" in respuesta.lower():
                registrar_evento(f"El usuario ha confirmado la toma de {med_name}.")
                veces = -1
            else:
                registrar_evento(f"El usuario no ha confirmado la toma de {med_name}.")
                veces += 1

            index = avisos.index(aviso)
            avisos[index] = (med_name, now, veces)
                

def ejecutar_programa_principal():
    while True:
        try:
            verificar_tomas()
            verificar_medicamentos()
            
            keyword_detected, message = esperando_clave()
            if keyword_detected:
                print("\nMensaje recibido después de la palabra clave")
                while True:
                    if message:
                        print("Usuario: " + message)
                        data = {"query": "La hora actual es: " + horaActual() + ". Respondeme a lo siguiente: " + message}
                        print("Generando respuesta...")
                        response = requests.post(url, json=data)
                        if response.status_code == 200:
                            response_data = response.json()
                            print(f"Respuesta: {response_data}")
                            talk(response_data)
                        else:
                            print(f"Error al generar respuesta: Código de estado {response.status_code}")
                            talk(f"Error al generar respuesta: Código de estado {response.status_code}")

                        message = ""

                    print("Esperando 5 segundos para otra pregunta...")
                    message = conversacion_fluida()
                    if not message:
                        break

            else:
                print("Palabra clave no detectada, reintentando...")
        except Exception as e:
            print(f"Error en el programa principal: {e}")

def main():
    try:
        borrarBD()
        cargarMedicamentos()

        ejecutar_programa_principal()
    except Exception as e:
        print(f"Error en el hilo principal: {e}")

if __name__ == "__main__":
    main()
