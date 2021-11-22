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

## References

https://arxiv.org/abs/1708.08988

https://github.com/cynricfu/multi-band-blending

[Computer Vision for Visual Effects](https://ocw.nthu.edu.tw/ocw/upload/125/1466/%E9%99%B3%E7%85%A5%E5%AE%97%20%E9%9B%BB%E8%85%A6%E8%A6%96%E8%A6%BA%E7%89%B9%E6%95%88%E2%96%95%20%E7%AC%AC%E4%B8%89%E5%91%A8%E8%AA%B2%E7%A8%8B%E8%B3%87%E6%96%99_TextureTransfer.pdf)
