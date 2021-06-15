# Desarrollo del ejercicio n° 1:

""" No supe aplicar muy bien la utilización del módulo sys como se mostraba en el código original. Encontré 
info relativa al código original en Stack Overflow (https://stackoverflow.com/questions/31689100/sys-argv1-indexerror-list-index-out-of-range)
pero tampoco logré destrabar esto. Busqué otros artículos y logré otra forma a través de la importación
del módulo NumPy complementario a cv2, que este último tiene la capacidad de procesar la imagen y el np 
(o numpy) tiene la capacidad de almacenar datos."""

# 1) Importar cv2 y numpy as np
import cv2
import numpy as np

# 2) A la variable faceCascade, le intregé el .xml directamente al argumento del clasificador, ensayé varias 
# opciones y esta fue la que me resultó.
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

# 3) 
""" Si bien no es muy eficiente, a continuación se definen 4 variables showImg para referenciar 4 documentos
.jpg. 

Acá se lee la imagen (favor marcar/desmarcar la línea de la variable showImg a utilizar y su 
respectiva aplicación em la variable image). Se mantienen las variables de image pero se cambia el argumento
en relación a la variable imagen que se quiera mostrar (modo prueba). """

# showImg1 = "image1.jpg" # No se puede utilizar un string o un arreglo para referenciar una variable
# showImg2 = "image2.jpg"
# showImg3 = "image3.jpg"
showImg4 = "image4.jpg"
# image = cv2.imread(showImg1) # Tuve problemas con la ruta /img así que moví la imagen a la misma carpeta del py y del xml
# image = cv2.imread(showImg2)
# image = cv2.imread(showImg3)
image = cv2.imread(showImg4)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Detección de grises

# 4) Detección de rostros o caras en las imágenes

faces = faceCascade.detectMultiScale(
	gray,
    scaleFactor=1.1,
    minNeighbors=9,
    minSize=(30, 30),
    maxSize=(200, 200)
    # flags = cv2.cv.CV_HAAR_SCALE_IMAGE    # Lo comenté porque me arrojaba un error en la ejecución que al 
                                            # momento de desmarcar, el código de reconocimiento funcionó bien.
)

# 5) Imprime o muestra la cantidad de rostros encontrados 

print("Found {0} faces!".format(len(faces)))

# 6) Dibuja el área del rectángulo en torno a los rostros, muestra la imagen, mantiene una espera indefinida
# sobre el resultado.
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


