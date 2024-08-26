# Usa la imagen base de Python más reciente
FROM python:latest

# Etiqueta del mantenedor
LABEL Maintainer="iblascoh"
# Establece el directorio de trabajo dentro del contenedor
WORKDIR /usr/app/src

# Copia todos los archivos Python y requirements.txt desde la carpeta 'code' al directorio de trabajo en el contenedor
COPY code/*.py ./
COPY requirements.txt ./

# Instala las dependencias especificadas en requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Instrucciones para construir y ejecutar la imagen
# 'docker image build --tag <tag nombre> .' #Construir la imagen
# 'docker image ls' #Comprobar que se ha creado la imagen
# 'docker run --rm <tag nombre> python bot.py param1, param2, param3, ...' #Ejecutar el contenedor
