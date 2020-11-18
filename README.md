# Seq_nms_YOLO

#### Membres: Yunyun SUN, Yutong YAN, Sixiang XU, Heng ZHANG

---

## Introduction

![](img/index.jpg) 

This project combines **YOLOv2**([reference](https://arxiv.org/abs/1506.02640)) and **seq-nms**([reference](https://arxiv.org/abs/1602.08465)) to realise **real time video detection**.

## Steps
1. Una vez descargado el código, deberás crear un entorno virtual con python 3.6. Para ello deberás irte al directorio /opt/anaconda3.7 y ejecutar:
`conda create --name envname python=3.6`;
`conda activate envname`;

1. Vamos a ir al archivo MakeFile, dentro de la carpeta seq\_yolo\_nms decargada y vamos a descativar los flags de OPENCV y CUDDN poniendo a '0' estas variables. A continuación, en el mismo MakeFile,  debemos cambiar las rutas `COMMON+= -DGPU -I/usr/local/cuda-8.0/include/` y `LDFLAGS+= -L/usr/local/cuda-8.0/lib64 -lcuda -lcudart -lcublas –lcurand`  por  `COMMON+= -DGPU -I/usr/local/cuda-10.1/include/`  y `LDFLAGS+= -L/usr/local/cuda-10.1/lib64 -lcuda -lcudart -lcublas –lcurand.`;

1. A continuación introduces por terminal: `export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}` 

1. `make` the project;

1. Descargamos los pesos para los modelos intruduciendo los siguientes comandos: `wget https://pjreddie.com/media/files/yolo.weights` y `wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights`

1. 





1. Download `yolo.weights` and `tiny-yolo.weights` by running `wget https://pjreddie.com/media/files/yolo.weights` and `wget https://pjreddie.com/media/files/tiny-yolo-voc.weights`;
1. Copy a video file to the video folder, for example, `input.mp4`;
1. In the video folder, run `python video2img.py -i input.mp4` and then `python get_pkllist.py`;
1. Return to root floder and run `python yolo_seqnms.py` to generate output images in `video/output`;
1. If you want to reconstruct a video from these output images, you can go to the video folder and run `python img2video.py -i output`

And you will see detection results in `video/output`

## Reference

This project copies lots of code from [darknet](https://github.com/pjreddie/darknet) , [Seq-NMS](https://github.com/lrghust/Seq-NMS) and  [models](https://github.com/tensorflow/models).
