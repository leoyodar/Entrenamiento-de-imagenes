from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.fftpack as fp
import tensorflow as tf
from tensorflow import keras
import pathlib

ventana = Tk()
ventana.title("Proyecto (Tercer Parcial)")
ventana.geometry("1400x600")

def cargar():
    cargada = filedialog.askopenfilename(title = "Selecciona Imagen", filetypes = (("all files","*.*"), ("all files","*.*")))
    return cargada

xxi=0
def mostrar():
    global xxi
    xxi = cargar()
    
    img = Image.open(xxi)
    saveimg = img
    A = np.array(img)
    
    width, height = img.size
             
    img = img.resize((400,400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(ventana, image=img,borderwidth=0)
    panel.image = img
    panel.place(x=100,y=168)
    prediccion()
    
porc=0
tip=0
def prediccion():
    global xxi,porc,tip
    tf.keras.backend.clear_session()
    
    data_dir = 'raw-img'
    
    batch_size = 32
    img_height = 180
    img_width = 180
    
    class_names = ['ardilla','caballo','vaca']
    print(class_names)
    
    prueba_path = xxi;
    
    img = tf.keras.utils.load_img(
        prueba_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    model = keras.models.load_model('modelo.h5')
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    porc=round(100 * np.max(score), 2) 
    tip=class_names[np.argmax(score)]
    
    subtituloimg1 = Label(ventana, text="Original", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=165, y=84)
    xtipoimg1 = Label(ventana, text=tip, font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=140, y=110)
    xporcentajeimg1 = Label(ventana, text=str(porc)+"%", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=190, y=136)
    
    #print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    
    
def contorno():
    global xxi,porc,tip
    
    img = Image.open(xxi)
    width, height = img.size
    
    for x in range(width):
        for y in range(height):
            r,g,b = img.getpixel((x,y))
            escala_gris = int(r * 0.299) + int(g * 0.587) + int(b * 0.114)
            pix = tuple([escala_gris,escala_gris,escala_gris])
            img.putpixel((x,y),pix)
    
    A = np.array(img)/255
    B = np.zeros(A.shape)
    s = A.shape

    r = np.array([[-1,-1,-1],
                  [-1,8,-1],
                  [-1,-1,-1]])
    
    for x in range(height-1):
        for y in range(width-1):
            B[x,y] = abs(r[0,0]*A[x-1, y-1]+         r[0,1]*A[x-1, y]+         r[0,2]*A[x-1, y+1]
                        +r[1,0]*A[x  , y-1]+         r[1,1]*A[x  , y]+         r[1,2]*A[x  , y+1]
                        +r[2,0]*A[x+1, y+1]+         r[2,1]*A[x+1, y]+         r[2,2]*A[x+1, y+1])
    
    img = Image.fromarray(np.uint8(B)*255)
    img = img.convert("L")
    
    img = img.resize((400,400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(ventana, image=img,borderwidth=0)
    panel.image = img
    panel.place(x=600,y=168)
    subtituloimg2 = Label(ventana, text="Contorno                             ", font=("Montserrat", 12), bg='#f7ad7b', fg='#ffffff').place(x=670, y=84)
    xtipoimg2 = Label(ventana, text=tip, font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=640, y=110)
    xporcentajeimg2 = Label(ventana, text=str(porc)+"%", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=690, y=136)

def laplaciano():
    global xxi,porc,tip
    
    img = Image.open(xxi)
    width, height = img.size
    
    for x in range(width):
        for y in range(height):
            r,g,b = img.getpixel((x,y))
            escala_gris = int(r * 0.299) + int(g * 0.587) + int(b * 0.114)
            pix = tuple([escala_gris,escala_gris,escala_gris])
            img.putpixel((x,y),pix)
    
    A = np.array(img)
    B = np.zeros(A.shape)
    width, height = img.size
    
    for x in range(width):
        for y in range(height):
            r,g,b = img.getpixel((x,y))
            escala_gris = int(r * 0.299) + int(g * 0.587) + int(b * 0.114)
            pix = tuple([escala_gris,escala_gris,escala_gris])
            img.putpixel((x,y),pix)
    
    r = np.array([[0,1,0],
                  [1,-4,1],
                  [0,1,0]])
    
    for x in range(height-1):
        for y in range(width-1):
            B[x,y] = abs(r[0,0]*A[x-1, y-1]+         r[0,1]*A[x-1, y]+         r[0,2]*A[x-1, y+1]
                        +r[1,0]*A[x  , y-1]+         r[1,1]*A[x  , y]+         r[1,2]*A[x  , y+1]
                        +r[2,0]*A[x+1, y+1]+         r[2,1]*A[x+1, y]+         r[2,2]*A[x+1, y+1])
 
    img = Image.fromarray(np.uint8(B))
    img = img.convert("L")
    
    img = img.resize((400,400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(ventana, image=img,borderwidth=0)
    panel.image = img
    panel.place(x=600,y=168)
    subtituloimg2 = Label(ventana, text="Laplaciano                             ", font=("Montserrat", 12), bg='#f7ad7b', fg='#ffffff').place(x=670, y=84)
    xtipoimg2 = Label(ventana, text=tip, font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=640, y=110)
    xporcentajeimg2 = Label(ventana, text=str(porc)+"%", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=690, y=136)
    
def suavizado_lineal():  
    global xxi,porc,tip
    
    img = Image.open(xxi)
    width, height = img.size
    
    A = np.array(img)/255
    B = np.zeros(A.shape)
    
    width, height = img.size

    r = 1/9
    
    for x in range(height-1):
        for y in range(width-1):
            B[x,y] = abs(r*A[x-1, y-1]+         r*A[x, y-1]+         r*A[x+1, y-1]
                        +r*A[x-1  , y]+         r*A[x  , y]+         r*A[x+1  , y]
                        +r*A[x-1, y+1]+         r*A[x, y+1]+         r*A[x+1, y+1])
 
    img = Image.fromarray(np.uint8((B)*255))
    img = img.resize((400,400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(ventana, image=img,borderwidth=0)
    panel.image = img
    panel.place(x=600,y=168)
    subtituloimg2 = Label(ventana, text="Suavizado Lineal                             ", font=("Montserrat", 12), bg='#f7ad7b', fg='#ffffff').place(x=670, y=84)
    xtipoimg2 = Label(ventana, text=tip, font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=640, y=110)
    xporcentajeimg2 = Label(ventana, text=str(porc)+"%", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=690, y=136)

def suavizado_gaussiano():
    global xxi,porc,tip
    
    img = Image.open(xxi)
    width, height = img.size
    
    A = np.array(img)/255
    B = np.zeros(A.shape)
    
    r = np.array([[1,2,1],
                  [2,4,2],
                  [1,2,1]])
    
    r = r*(1/16)
    
    for x in range(height-1):
        for y in range(width-1):
            B[x,y] = abs(r[0,0]*A[x-1, y-1]+         r[0,1]*A[x-1, y]+         r[0,2]*A[x-1, y+1]
                        +r[1,0]*A[x  , y-1]+         r[1,1]*A[x  , y]+         r[1,2]*A[x  , y+1]
                        +r[2,0]*A[x+1, y+1]+         r[2,1]*A[x+1, y]+         r[2,2]*A[x+1, y+1])
 
    img = Image.fromarray(np.uint8((B)*255))
    img = img.resize((400,400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(ventana, image=img,borderwidth=0)
    panel.image = img
    panel.place(x=600,y=168)
    subtituloimg2 = Label(ventana, text="Suavizado Gaussiano                             ", font=("Montserrat", 12), bg='#f7ad7b', fg='#ffffff').place(x=670, y=84)
    xtipoimg2 = Label(ventana, text=tip, font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=640, y=110)
    xporcentajeimg2 = Label(ventana, text=str(porc)+"%", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=690, y=136)
    
def fourier():
    global xxi,porc,tip
    
    img = cv2.imread(xxi,0)
 
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
     
    magnitudFFT = 20*np.log(np.abs(fshift))
    
    imagen = Image.fromarray(np.uint8(magnitudFFT))   
    imagen = imagen.resize((400,400), Image.ANTIALIAS)
    imagen.save("Fourier.jpg")
    
    fou = Image.open("Fourier.jpg")
    fou = ImageTk.PhotoImage(fou)
    
    panel = Label(ventana, image=fou,borderwidth=0)
    panel.image = fou
    panel.place(x=600,y=168)
    subtituloimg2 = Label(ventana, text="Fourier                             ", font=("Montserrat", 12), bg='#f7ad7b', fg='#ffffff').place(x=670, y=84)
    xtipoimg2 = Label(ventana, text=tip, font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=640, y=110)
    xporcentajeimg2 = Label(ventana, text=str(porc)+"%", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=690, y=136)

def paso_alto():
    global xxi,porc,tip
    img = cv2.imread(xxi,0)

    f = np.fft.fft2(img)
    fpa = np.fft.fftshift(f)

    width, height = img.shape
    cwidth,cheight = int(width/2) , int(height/2)
    fpa[cwidth-25:cwidth+25+1, cheight-25:cheight+25+1] = 0
    pa = fp.ifft2(fp.ifftshift(fpa)).real
    
    img = Image.fromarray(np.uint8((pa))) 
    imagen = Image.fromarray(np.uint8((pa)))  
    
    img = img.resize((400,400), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(ventana, image=img,borderwidth=0)
    panel.image = img
    panel.place(x=600,y=168)
    
    subtituloimg2 = Label(ventana, text="Paso Alto                            ", font=("Montserrat", 12), bg='#f7ad7b', fg='#ffffff').place(x=670, y=84) 
    xtipoimg2 = Label(ventana, text=tip, font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=640, y=110)
    xporcentajeimg2 = Label(ventana, text=str(porc)+"%", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=690, y=136)
    
fondo = Label(ventana, text="", bg='#f7f7f7', padx = 1000, pady =300).place(x=0, y=0)
fondo2 = Label(ventana, text="", bg='#B7D4E7', padx = 200, pady =300).place(x=1120, y=0)
fondoimg1 = Label(ventana, text="", bg='#f7ad7b', padx = 208, pady =237).place(x=90, y=84)
fondoimg2 = Label(ventana, text="", bg='#f7ad7b', padx = 208, pady =237).place(x=590, y=84)

tituloimg1 = Label(ventana, text="Imagen:", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=100, y=84)
tipoimg1 = Label(ventana, text="Tipo:", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=100, y=110)
porcentajeimg1 = Label(ventana, text="Porcentaje:", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=100, y=135)
tituloimg2 = Label(ventana, text="Imagen:", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=600, y=84)
tipoimg2 = Label(ventana, text="Tipo:", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=600, y=110)
porcentajeimg2 = Label(ventana, text="Porcentaje:", font=("Montserrat", 11), bg='#f7ad7b', fg='#ffffff').place(x=600, y=135)


btn = Button(ventana, text="Insertar Imagen", command=mostrar, bg='#ff6424', fg='#ffffff', relief=FLAT, width='26', height='2', cursor="hand2", font=("Montserrat", 9)).place(x=20, y=18)
btn2 = Button(ventana, text="Contorno", command=contorno, bg='#3368BA', fg='#ffffff', relief=FLAT, width='26', height='2', cursor="hand2", font=("Montserrat", 9)).place(x=1150, y=20)
btn3 = Button(ventana, text="Laplaciano", command=laplaciano, bg='#3368BA', fg='#ffffff', relief=FLAT, width='26', height='2', cursor="hand2", font=("Montserrat", 9)).place(x=1150, y=120)
btn4 = Button(ventana, text="Suavizado Lineal", command=suavizado_lineal, bg='#3368BA', fg='#ffffff', relief=FLAT, width='26', height='2', cursor="hand2", font=("Montserrat", 9)).place(x=1150, y=220)
btn5 = Button(ventana, text="Suavizado Gaussiano", command=suavizado_gaussiano, bg='#3368BA', fg='#ffffff', relief=FLAT, width='26', height='2', cursor="hand2", font=("Montserrat", 9)).place(x=1150, y=320)
btn6 = Button(ventana, text="Fourier", command=fourier, bg='#3368BA', fg='#ffffff', relief=FLAT, width='26', height='2', cursor="hand2", font=("Montserrat", 9)).place(x=1150, y=420)
btn7 = Button(ventana, text="Frecuencial", command=paso_alto, bg='#3368BA', fg='#ffffff', relief=FLAT, width='26', height='2', cursor="hand2", font=("Montserrat", 9)).place(x=1150, y=520)

ventana.mainloop()
