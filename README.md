# MoveNet-Python-Example

:hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster:  
**This is Fork for checking the operation of MoveNet, which has been modified independently.**  
:hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster::hamster:  

- MoveNet MultPose + 10 person special + USB Camera 480x640

https://user-images.githubusercontent.com/33194443/190292838-bdc38f2c-d3e5-4d54-96b4-51994fce3e95.mp4

https://user-images.githubusercontent.com/33194443/190295137-ca6194cc-6b41-44be-85c0-d0dc0c4bffd4.mp4

https://user-images.githubusercontent.com/33194443/190524104-bf56a9c1-2d60-482f-af4a-342c5f570845.mp4

- PoseBase Hand-Crop - Flip image left/right

https://user-images.githubusercontent.com/33194443/190611783-24c5c671-005e-4a65-8e34-5cc81ff33a62.mp4

- PoseBase Hand-Crop - Flip image left/right - Palm Detection + Rotation

https://user-images.githubusercontent.com/33194443/191143828-3983a512-4665-495e-9f87-702b4690d3c3.mp4

- PoseBase Hand-Crop - Flip image left/right - Palm Detection + Rotation - HandLandmark Detection

https://user-images.githubusercontent.com/33194443/191156009-f4a481d5-ec84-4581-80ee-542c32f34e6a.mp4

---

[MoveNet](https://tfhub.dev/s?q=MoveNet)のPythonでの動作サンプルです。<br>
ONNXに変換したモデルも同梱しています。変換自体を試したい方は[MoveNet_tf2onnx.ipynb](MoveNet_tf2onnx.ipynb)を使用ください。<br>

![smjqx-4ndt8](https://user-images.githubusercontent.com/37477845/130482531-5be5f3e6-0dc9-42bb-80a8-4e7544d9ba7e.gif)

2021/08/24時点でTensorFlow Hubで提供されている以下モデルを使用しています。
* [movenet/singlepose/lightning(v4)](https://tfhub.dev/google/movenet/singlepose/lightning/4)
* [movenet/singlepose/thunder(v4)](https://tfhub.dev/google/movenet/singlepose/thunder/4)
* [movenet/multipose/lightning(v1)](https://tfhub.dev/google/movenet/multipose/lightning/1)

# Requirement 
* TensorFlow 2.3.0 or later
* tensorflow-hub 0.12.0 or later
* OpenCV 3.4.2 or later
* onnxruntime 1.5.2 or later ※ONNX推論を使用する場合のみ


# Demo
デモの実行方法は以下です。
#### SignlePose
```bash
python demo_singlepose.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --file<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --mirror<br>
VideoCapture()取り込みデータを左右反転するか否か<br>
デフォルト：指定なし
* --model_select<br>
使用モデルの選択<br>
Saved Model, ONNX：0→Lightning　1→Thunder<br>
TFLite：0→Lightning(float16)　1→Thunder(float16)　2→Lightning(int8)　3→Thunder(int8)<br>
デフォルト：0
* --keypoint_score<br>
キーポイント表示の閾値<br>
デフォルト：0.4

#### MultiPose
```bash
python demo_multipose.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --file<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --mirror<br>
VideoCapture()取り込みデータを左右反転するか否か<br>
デフォルト：指定なし
* --keypoint_score<br>
キーポイント表示の閾値<br>
デフォルト：0.4
* --bbox_score<br>
バウンディングボックス表示の閾値<br>
デフォルト：0.2

# Reference
* [TensorFlow Hub：MoveNet](https://tfhub.dev/s?q=MoveNet)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
MoveNet-Python-Example is under [Apache-2.0 License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[ストリートバスケット](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002080169_00000)を使用しています。
