# Plate-identifier
This project was created using Open CV on Python, as well as the library PyTesseract to capture the plate of the car and HSV to identify the color of the plate.

import cv2
import numpy as np
from PIL import Image
import pytesseract

# Ruta del ejecutable de Tesseract OCR / Executer route of Tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'

# Cargar el video / Upload the video / You need to upload the video on the same charpet of file where is the file.py to use the video as well.
cap = cv2.VideoCapture("Test7.mp4")
Ctexto = ''

while True:
    # Leer el fotograma del video / Read the frames of the video. 
    ret, frame = cap.read()

    if ret == False:
        break

    # Obtener dimensiones del fotograma / Get the dimension of the frames
    al, an, c = frame.shape

    # Definir las coordenadas de la zona de interés / define the coordinates of the interest area. 
    x1 = int(an / 5) - 30 # Desplazamiento hacia la izquierda
    x2 = int(an * 2 / 5) + 320  # Desplazamiento hacia la izquierda
    y1 = int(al / 3)
    y2 = int(al * 2 / 3)

    # Dibujar rectángulo para la zona de interés / Draw a rectangle of the interest area.
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Recortar la zona de interés / Reduce the interest area
    roi = frame[y1:y2, x1:x2]

    # Convertir a espacio de color HSV / convert the color RGB.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Definir el rango de color para la placa amarilla / define the range of the lower and upper of the color yellow from RGB to HSV.
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)

    # Aplicar máscara para obtener solo píxeles amarillos / Apply the mask to get the yellow pixels
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Encontrar contornos de las regiones amarillas / Find the contours of the yellow regions.
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if 500 < area < 5000:
            # Obtener coordenadas del rectángulo del contorno / get the coordinates of the rectangle contours.
            x, y, width, height = cv2.boundingRect(contour)

            # Coordenadas de la placa en el fotograma original / Coordinates of the plates with the original frames.
            xpi = x + x1
            ypi = y + y1
            xpf = x + width + x1
            ypf = y + height + y1

            # Dibujar rectángulo alrededor de la placa / Draw the rectangle around of the plate.
            cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (0, 255, 0), 2)

            # Recortar la placa / Cut the plate.
            plate = frame[ypi:ypf, xpi:xpf]

            # Convertir a escala de grises / Convert to grayscale.
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

            # Binarizar la imagen en color negro / Binarize the picture to black color.
            _, binary_plate = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Convertir matriz en imagen / Convert the matrix to image.
            binary_plate = Image.fromarray(binary_plate)
            binary_plate = binary_plate.convert("L")

            # Verificar el tamaño de la placa / verif the size of the plate
            if height >= 36 and width >= 82:
                # Extraer el texto de la placa / get the characteristics of the plate
                config = "--psm 7"  # Modo 7 para placas con 7 caracteres / here you can change the size of character 
                text = pytesseract.image_to_string(binary_plate, config=config)

                # Verificar si el texto es válido
                if len(text) >= 7:
                    Ctexto = text[0:7]
                    print(Ctexto)

                    # Dibujar un rectángulo y mostrar el texto
                    cv2.rectangle(frame, (870, 750), (1070, 850), (0, 0, 0), cv2.FILLED)
                    cv2.putText(frame, Ctexto[0:7], (900, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el fotograma en una ventana
    cv2.imshow("Vehiculos", frame)

    # Leer tecla
    key = cv2.waitKey(40)
    if key == 27:  # Si se presiona Esc, salir del bucle
        break

# Liberar el video y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
