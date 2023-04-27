from compreface import CompreFace
from compreface.service import RecognitionService
from compreface.collections import FaceCollection
from compreface.collections.face_collections import Subjects

DOMAIN: str = 'http://localhost'
PORT: str = '8000'
API_KEY: str = '00000000-0000-0000-0000-000000000002'

compre_face: CompreFace = CompreFace(DOMAIN, PORT)

recognition: RecognitionService = compre_face.init_face_recognition(API_KEY)

face_collection: FaceCollection = recognition.get_face_collection()

subjects: Subjects = recognition.get_subjects()

image_path: str = '2.jpg'
image_path: str = '3.jpg'

print(recognition.recognize(image_path=image_path))
print(recognition.recognize(image_path=image_path))