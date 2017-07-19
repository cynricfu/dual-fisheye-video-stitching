# SummerResearch

## Introduction

A summer research project to seamlessly stitch dual-fisheye video into 360-degree videos.

## Dependencies

* Python 2.7+
* OpenCV 3.2.0

## Run

```
python demo.py [-h] [-o OUTPUT.XYZ] INPUT.XYZ
```

```
positional arguments:
  INPUT.XYZ             path to the input dual fisheye video

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT.XYZ, --output OUTPUT.XYZ
                        path to the output equirectangular video
```

## References

http://www.kscottz.com/fish-eye-lens-dewarping-and-panorama-stiching/

https://trac.ffmpeg.org/wiki/RemapFilter

http://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

https://support.google.com/youtube/answer/6178631?hl=en
