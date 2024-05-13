import json
import time
import requests
import whisper
import pyttsx3
import speech_recognition as sr
import shutil
import os
import tempfile
import datetime

url = "http://localhost:8080/ask"
urlDoc= "http://localhost:8080/docs"

file = tempfile.mkdtemp()
path = os.path.join(file, 'temp.wav')

listener = sr.Recognizer()

engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty('rate', 145)
engine.setProperty('voice', voices[0].id)

def talk(text):
    engine.say(text)
    engine.runAndWait()

def listen_once():
    with sr.Microphone() as source:
        print("Esperando palabra clave...")
        listener.adjust_for_ambient_noise(source)
        try:
            audio = listener.listen(source, timeout=5, phrase_time_limit=5)
            detected_text = listener.recognize_google(audio, language='es-ES')
            if 'ordenador' in detected_text.lower():
                print("Palabra clave detectada. Escuchando...")
                return True, detected_text[detected_text.lower().index('ordenador') + len('ordenador'):].strip()
            else:
                return False, ""
        except sr.UnknownValueError:
            print("No se pudo entender el audio, intentando de nuevo...")
        except sr.RequestError as e:
            print(f"Error de la API de reconocimiento de voz; {e}")
        except sr.WaitTimeoutError:
            print("No se detectó ninguna entrada de voz.")
    return False, ""



def recognize_audio(path):
    model = whisper.load_model("base")
    transcription = model.transcribe(path, language="spanish", fp16=False)
    return transcription["text"]

def cambiarHora():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M")
    filepath = 'docs/time.txt'

    with open(filepath, 'w') as archivo:
        archivo.write("Hora actual: " + current_time)

    with open(filepath, 'rb') as archivo:
        files = {"file": archivo}
        response = requests.post(urlDoc, files=files)

        if response.status_code == 200:
            print("Cambio de hora realizado")
        else:
            print(f"Error al enviar archivo: Código de estado {response.status_code}")

    return current_time

def check_medication_times(medications, current_time):
    for med in medications:
        name = med['name']
        times = med['hours'].split(', ')
        for t in times:
            if t == current_time:
                print(f"Es hora de tomar tu medicamento: {name} ({med['brand']})")

def cargarMedicamentos():
    with open('docs/medicamentos.txt', 'rb') as archivo:
        files = {"file": archivo}
        response = requests.post(urlDoc, files=files)

        if response.status_code == 200:
            print("Medicamentos cargados")
        else:
            print(f"Error al enviar archivo: Código de estado {response.status_code}")

def borrarBD():
    folder="db"
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

def main():
    borrarBD()
    current_time = cambiarHora()
    cargarMedicamentos()

    while True:
        
        if current_time != datetime.datetime.now().strftime("%H:%M"):
            current_time = cambiarHora()

        with open('docs/medicamentos.txt', 'r') as file:
            medicamentos_data = json.load(file)

        check_medication_times(medicamentos_data['medications'], current_time)
        
        keyword_detected, message = listen_once()
        if keyword_detected:
            print("\nMensaje recibido después de la palabra clave")
            if message:
                print("Usuario: " + message)
                if message.lower().__contains__('terminar'):
                    talk("Terminar")
                    break
                else:
                    data = {"query": message}
                    print("Generando respuesta...")
                    response = requests.post(url, json=data)
                    if response.status_code == 200:
                        response_data = response.json()
                        print(f"Respuesta: {response_data}")
                        talk(response_data)
                    else:
                        print(f"Error al generar respuesta: Código de estado {response.status_code}")
                        talk(f"Error al generar respuesta: Código de estado {response.status_code}")
        else:
            print("Palabra clave no detectada, reintentando...")
            time.sleep(1)

if __name__ == "__main__":
    main()
