# SummerResearch

A summer research project to seamlessly stitch dual-fisheye video into 360-degree videos using OpenCV-Python.

## Dependencies

* Python 2.7+
* OpenCV 3.3.0

## Run

```
python main.py [-h] [-o OUTPUT.XYZ] INPUT.XYZ
```

```
positional arguments:
  INPUT.XYZ             path to the input dual fisheye video

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT.XYZ, --output OUTPUT.XYZ
                        path to the output equirectangular video
```
