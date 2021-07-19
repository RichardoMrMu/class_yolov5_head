# class_yolov5_head
get head bbox by yolov5 in class, which can count class student number and get student whether uphead or downhead
## Introduction

class_Yolov5_head_ is a real-time,high accuracy head detection and can classify head pose including uphead and downhead.

## Data preparation
1. get your own data including *.jpg and xml files(label).
2. use ./data/retinaface2yolo.py to get yolo training dataset from xml files.
```python
python ./data/retinaface2yolo.py
python ./data/val2yolo.py
```


## Training 
1. you can get pretrain model from [yolov5-face](https://github.com/deepcam-cn/yolov5-face), and you should set path in train.py. You should set `weights` as your default pretrain model, set `cfg` as your yaml file from yolov5-face yaml files.
2. you can set `epoch` and `batch_size` by all you want.

## WIDERFace Evaluation

```shell
python3 test_widerface.py --weights 'your test model' --img-size 640

cd widerface_evaluate
python3 evaluation.py
```

## Test 
1. you can use `detect_face.py` to test your model, by images or videos.

#### References

https://github.com/ultralytics/yolov5

https://github.com/DayBreak-u/yolo-face-with-landmark

https://github.com/xialuxi/yolov5_face_landmark

https://github.com/biubug6/Pytorch_Retinaface

https://github.com/deepinsight/insightface
