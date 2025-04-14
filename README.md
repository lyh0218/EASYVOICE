基于F5-TTS开发的推理项目

Python的版本为3.10.11

第一次启动项目会从hugging face上下载模型，可能需要科学上网

需要安装并配置CUDA12.4

需要安装并配置ffmpeg或者在data/ffmpeg目录下放入ffmpeg.exe ffplay.exe ffprobe.exe

需要在data/whisper_models目录下放入whisper模型 [模型下载](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt) 除非你用不到相关功能

需要在data/tts_models目录下放入训练好的F5-TTS-V1模型

使用 
```Bash
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```
安装Pytroch，一定要先配置CUDA再安装Pytroch，这点很重要

启动app.py来启动程序 打开 http://127.0.0.1:7860/ 访问WebUI
