import cv2
import datetime
import mysql.connector

# Charger le classificateur Haar cascade pour la détection des visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Connect to the database
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="teamdev"
)

# Définir une fonction pour détecter les visages et ajouter l'heure de pointage
def detect_and_mark_attendance(img):
    # Convertir l'image en niveaux de gris pour une meilleure détection des visages
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    # Ajouter l'heure de pointage actuelle pour chaque visage détecté et insérer les données dans la base
    for (x, y, w, h) in faces:
        # Dessiner un rectangle autour du visage détecté
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Ajouter l'heure de pointage actuelle sous le visage détecté
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, timestamp, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Insérer les données dans la base de données
        mycursor = mydb.cursor()
        sql = "INSERT INTO pointage (`date`,`heure_pointage`,`utilisateur_id`) VALUES (%s, %s, %s)"
        val = (timestamp, timestamp, "103")
        mycursor.execute(sql, val)
        mydb.commit()
    return img

# Ouvrir la webcam pour capturer la vidéo en direct
cap = cv2.VideoCapture(0)

while True:
    # Lire une image de la webcam
    ret, img = cap.read()
    # Détecter les visages et ajouter l'heure de pointage à l'image
    img = detect_and_mark_attendance(img)
    # Afficher l'image avec les visages détectés et les heures de pointage correspondantes
    cv2.imshow('Attendance', img)
    # Attendre la touche 'q' pour quitter la boucle de capture vidéo
    if cv2.waitKey(1) == ord('q'):
        break

# Fermer la fenêtre d'affichage et libérer la ressource de capture vidéo
cap.release()
cv2.destroyAllWindows()
