import cv2
from PIL import Image
import numpy as np

def is_bw_or_grayscale(image_path):
    """
    Check if the image is black and white or grayscale
    """
    image = Image.open(image_path)
    image = image.convert('RGB')
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            if r != g != b:
                return False
    return True

def is_blue(color):
    blue, green, red = color
    print(blue,green,red,color)
    return blue > 100 and green < 100 and red < 100

def main(image_path):
    # Check if the image is black and white or grayscale
    if is_bw_or_grayscale(image_path):
        print("Image is black and white or grayscale. Validation failed.")
        return

    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Calculate the total area of faces detected
    total_face_area = sum([w * h for x, y, w, h in faces])
    
    # Check the background color
    mean_color = np.mean(image.reshape(-1, image.shape[-1]), axis=0)
    if is_blue(mean_color):
        background_color = "blue"
    else:
        background_color = "not blue"

    # Calculate the percentage of the face area in the image
    image_area = image.shape[0] * image.shape[1]
    face_percentage = total_face_area / image_area

    # Determine if the image meets the criteria
    if face_percentage >= 0.2:
        print(f"Image validated:good to go. {round(face_percentage*100,2)}%")
    else:
        print(f"Image not validated: Does not meet the criteria. Face Coverage is only {round(face_percentage*100,2)}%")

if __name__ == "__main__":
    image_path = "b.jpg"  # Provide the path to your image
    main(image_path)
