from gettext import npgettext
import pyodbc
import cv2

# Connect to SQL database
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=server_name;DATABASE=teamdev;UID=root;PWD=')
cursor = conn.cursor()

# Retrieve image from database
cursor.execute("SELECT image_data FROM utilisateur WHERE id=1")
row = cursor.fetchone()

# Convert image data to numpy array
img_data = row[0]
img_array = np.frombuffer(img_data, npgettext.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)

# Close the database connection
cursor.close()
conn.close()
