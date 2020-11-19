# Seq_nms_YOLO

#### Membres: Yunyun SUN, Yutong YAN, Sixiang XU, Heng ZHANG

---

## Introduction

![](img/index.jpg) 

This project combines **YOLOv2**([reference](https://arxiv.org/abs/1506.02640)) and **seq-nms**([reference](https://arxiv.org/abs/1602.08465)) to realise **real time video detection**.

## Steps
1. Una vez descargado el código, deberás crear un entorno virtual con python 3.6. Para ello deberás irte al directorio /opt/anaconda3.7 y ejecutar: \
`conda create --name envname python=3.6`\
`conda activate envname`

1. Vamos a ir al archivo MakeFile, dentro de la carpeta seq\_yolo\_nms decargada y vamos a descativar los flags de OPENCV y CUDDN poniendo a '0' estas variables. A continuación, en el mismo MakeFile,  debemos cambiar las rutas `COMMON+= -DGPU -I/usr/local/cuda-8.0/include/` y `LDFLAGS+= -L/usr/local/cuda-8.0/lib64 -lcuda -lcudart -lcublas –lcurand`  por  `COMMON+= -DGPU -I/usr/local/cuda-10.1/include/`  y `LDFLAGS+= -L/usr/local/cuda-10.1/lib64 -lcuda -lcudart -lcublas –lcurand.`

1. A continuación introduces por terminal: `export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}` 

1. `make` the project

1. Descargamos los pesos para los modelos intruduciendo los siguientes comandos: `wget https://pjreddie.com/media/files/yolo.weights` y `wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights`

1. Copia un archivo de vídeo en la carpeta video, por ejemplo `input.mp4`

1. En el archivo video2img.py realizamos los siguientes cambios:\
-Poner paréntesis a los print de las líneas: 6 y 39

1. Decargamos la librería OpenCv con el siguiente comando: `conda install -c conda-forge opencv`

1. Desde la carpeta de vídeo ejecutamos el archivo video2img.py. Para ello introducimos por terminal: `python video2img.py -i input.mp4` y a continuación `python get_pkllist.py`

1. Volvemos a la carpeta yolo_seq_nms. En el archivo yolo_seqnms.py debemos realizar los siguientes cambios:\
-Poner los paréntesis a los print de las líneas: 46, 90, 104, 264, 270, 276, 290 y 292\
-cambiar el `import cPickle as pickle`por `import pickle`\
-En la línea 291 sustituímos `scipy.misc.imsave()` por `imageio.imwrite()` \
-Introducimmos `import imageio`

1. Instalamos el módulo Imageio. Para ello introducimos el siguiente comando: `conda install -c menpo imageio`

1. Instalamos la librería MatplotLib con el comando: `pip install matplotlib`

1. En el archivo yolo_detection.py realizamos los siguienets cambios:\
-Poner los paréntesis a los print de las líneas: 134, 135, 146, 160 y 161\
-Introducir un `import os`\
-Cambiar la línea 37 `(lib = CDLL("libdarknet.so", RTLD_GLOBAL))` por `lib = CDLL(os.path.join(os.getcwd(), "libdarknet.so"), RTLD_GLOBAL)`\
-Cambiar las líneas `net = load_net(cfg, weights, 0)`, `meta = load_meta(data)`  e `im = load_image(image, 0, 0)` por `net = load_net(bytes(cfg, 'utf-8')`, `bytes(weights, 'utf-8'), 0)`, `meta = load\_meta(bytes(data, 'utf-8'))` y `im = load\_image(bytes(image, 'utf-8'), 0, 0)`

1. Introducimos por terminal los siguientes export:\
`export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}`\
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64`\
`export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.1/lib64`\

1. Instalamos el módulo de object detection con el siguiente comando: `pip install tensorflow-object-detection-api`

1. En el archivo label_map_util.py realizamos los siguientes cambios:\
-En la línea 116 cambiar `tf.gfile.GFile(path, 'r')`  por `tf.io.gfile.GFile(path, 'r').`

1. Ejecutamos el archivo yolo_seqnms. Para ello introducimos por terminal: `python yolo_seqnms.py`. De esta forma se generan las frames en la carpeta `video/output`

1. En el archivo img2video.py hay que realizar los siguientes cambios:\
-Poner los paréntesis a los print de las líneas: 6, 13 y 30\
-Sustituir la línea 43 `if box[0]==cls` por `if box[0]==cls.encode('utf8')`

1. Para reconstruir el vídeo te vas a la carpeta de video y ejecutas:  `python img2video.py -i output`. Esto te va a generar un vídeo con nombre `output.mp4` en la carpeta de video. 

# Nota
Todas estas instrucciones estan hechas para ser realizadas en un ordenador del laboratorio 16 de la EPS-UAM.



## Reference

This project copies lots of code from [darknet](https://github.com/pjreddie/darknet) , [Seq-NMS](https://github.com/lrghust/Seq-NMS) and  [models](https://github.com/tensorflow/models).
