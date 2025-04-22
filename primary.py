import json
import os
import random
import time

from opencc import OpenCC

'''
    基础管理api
'''

# # 加载 Whisper 模型 (语音识别模型)
# def load_whisper_model(whisper_model_part):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = whisper.load_model(
#             whisper_model_part,  # 模型大小 : "tiny" "base" "small" "medium" "large"
#         device=device,
#         download_root=os.path.dirname("data/whisper_models")
#     )
#     return model
#
# # 加载 F5TTS（文本转语音 TTS 模型）
# def load_f5tts_model(f5tts_model_part):
#     return F5TTS(model = "F5TTS_v1_Base",  # 基础版模型
#                  ckpt_file = f5tts_model_part,  #
#                  vocab_file ="data/vocab.txt",
#                  ode_method="euler", use_ema=True,
#                  vocoder_local_path = None,
#                  device = None,
#                  hf_cache_dir ="data/huggingface_models")
# 音频转文本
def whisper_speech_to_text(model, audio_file_path):
    cc_model = OpenCC('t2s')
    if os.path.exists(audio_file_path):
        result = model.transcribe(audio_file_path,language = "zh",temperature = 0.2,beam_size = 5)["text"].strip()
    else:
        result = None
    return cc_model.convert(result)
# 获取本地语音文本集
def get_speaker_config_json():
    folder_path = "speaker/speaker_config"
    json_files = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                file_name_without_ext = os.path.splitext(file)[0]
                json_files.append(file_name_without_ext)
    return json_files
#根据声音名获取对应声音路径与文本
def get_speaker_config_value(file_name):
    file_path = f"speaker/speaker_config/{file_name}.json"
    try:
        with open(file_path, 'r',encoding='utf-8') as f:
            speaker_data = json.load(f)
            voice_path = speaker_data['voice_path']
            reference_text = speaker_data['reference_text']
            return voice_path, reference_text
    except Exception as e:
        print(f"Error opening speaker config file: {e}")
# 随机选择
def speaker_random_select(choice):
    return choice[random.randint(0, len(choice)-1)]
# 获取生成声音路径
def get_voice_path():
    file_paths = []
    for root, dirs, files in os.walk("output/output_voice"):
        for file in files:
            if not file.endswith('.gitkeep'):
                file_paths.append(f"{root}/{file}")
    return file_paths
#获取生成ppt路径
def get_pptx_path():
    file_paths = []
    for root, dirs, files in os.walk("output/output_pptx"):
        for file in files:
            if not file.endswith('.gitkeep'):
                file_paths.append(f"{root}/{file}")
    return file_paths
# 生成视频路径
def get_view_path():
    file_paths = []
    for root, dirs, files in os.walk("output/output_view"):
        for file in files:
            if not file.endswith('.gitkeep'):
                file_paths.append(f"{root}/{file}")
    return file_paths
#当前时间
def get_time():
        timestamp = time.time()
        local_time = time.localtime(timestamp)
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S",local_time) + f".{int((timestamp - int(timestamp)) * 100):02d}"
        return time_str