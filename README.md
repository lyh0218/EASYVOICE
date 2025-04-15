基于F5-TTS开发的推理项目

功能一 人声库 上传一段3~8秒的音频作为推理的参考音频

功能二 音频生成 核心功能，通过参考音频和输入文本生成音频

功能三 PPT制作 为PPT添加语音讲解

功能四 视频配音 上传一段视频可以对视频进行配音替换和添加字幕

Python的版本为3.10.11

使用
```bash
python.exe -m pip install --upgrade pip
```
更新你的pip

第一次启动项目会从hugging face上下载模型，可能需要科学上网

需要安装并配置CUDA12.4

需要安装并配置ffmpeg 或者 在data/ffmpeg目录下放入ffmpeg.exe ffplay.exe ffprobe.exe

需要在data/whisper_models目录下放入whisper模型 [medium模型下载](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt) [base模型下载](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)除非你用不到相关功能

需要在data/tts_models目录下放入训练好的F5-TTS-V1模型

使用 
```Bash
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```
或者改用清华大学源
```Bash
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu124
```
安装Pytroch，一定要先配置CUDA再安装Pytroch，这点很重要

使用
```Bash
pip install -r requirements.txt
```
或
```Bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
安装依赖

启动app.py来启动程序 打开 http://127.0.0.1:7860/ 访问WebUI
