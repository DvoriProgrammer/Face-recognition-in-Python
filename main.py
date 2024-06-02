import cv2
import sys, pathlib

# Define the paths to the image and the Haar cascade file
imagePath = r"C:\Users\user1\Desktop\pic\da.jpg"
cascader_file = str(pathlib.Path(__file__).parent / "haarcascade_frontalface_default.xml")

# Load the Haar cascade file
faceCascade = cv2.CascadeClassifier(cascader_file)

# Read the image
image = cv2.imread(imagePath)

# Check if the image was successfully loaded
if image is None:
    print("Error loading image")
    sys.exit()

# Resize the image
scale_percent = 45  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert the image to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(20, 20),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print(f"Found {len(faces)} faces!")

# Draw a rectangle around the faces and print the coordinates
for (x, y, w, h) in faces:
    cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # print(f"Face found at: x={x}, y={y}, width={w}, height={h}")

# Create a smaller window
cv2.namedWindow("Faces found", cv2.WINDOW_NORMAL)

# Display the output
cv2.imshow("Faces found", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()