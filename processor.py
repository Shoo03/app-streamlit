import pandas as pd
import numpy as np
import string

# Todos los caracteres posibles
caracteres_posibles = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

def extraer_features(etiqueta, color):
    caracteres_posibles = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
    longitud = len(etiqueta)
    vector_aparicion = pd.Series(0, index=caracteres_posibles)
    vector_posicion = pd.Series(0, index=caracteres_posibles)

    for i, letra in enumerate(etiqueta):
        if letra in caracteres_posibles:
            vector_aparicion[letra] = 1
            vector_posicion[letra] = i + 1

    features = pd.concat([
        pd.Series({'longitud': longitud, 'color': int(color)}),
        vector_aparicion.add_prefix('a_'),
        vector_posicion.add_prefix('p_')
    ])

    return features




def segmentar_letras(imagen_2d, texto):
    alto, ancho = imagen_2d.shape
    num_letras = len(texto)
    bloques = num_letras + 2  # añadimos margen a izquierda y derecha
    ancho_bloque = ancho // bloques

    letras = []
    for i in range(1, bloques - 1):  # omitimos primero (0) y último (n+1)
        inicio = i * ancho_bloque
        fin = (i + 1) * ancho_bloque
        subimg = imagen_2d[:, inicio:fin]
        letras.append((subimg, texto[i - 1]))  # i-1 para que coincida con letra real

    return letras
