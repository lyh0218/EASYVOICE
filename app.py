import os.path
ffmpeg_path = "data/ffmpeg/"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
import shutil
from f5_tts.api import F5TTS
import torch
import whisper
import web


# 加载 Whisper 模型 (语音识别模型)
def load_whisper_model(whisper_model_part):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(
            whisper_model_part,  # 模型大小 : "tiny" "base" "small" "medium" "large"
        device=device,
        download_root=os.path.dirname("data/whisper_models")
    )
    return model

# 加载 F5TTS（文本转语音 TTS 模型）
def load_f5tts_model(f5tts_model_part):
    return F5TTS(model = "F5TTS_v1_Base", # 基础版模型
                 ckpt_file = f5tts_model_part, #
                 vocab_file = "data/vocab.txt",
                 ode_method="euler",use_ema=True,
                 vocoder_local_path = None,
                 device = None,
                 hf_cache_dir = "data/huggingface_models")

def main():

    shutil.rmtree("data/temp") # 清除临时文件
    os.makedirs("data/temp")
    os.makedirs("data/temp/.gitkeep")
    tts_model = load_f5tts_model("data/tts_models/pro_v2.0.pt")
    whisper_model = load_whisper_model("data/whisper_models/medium.pt")

    web.creat_web(whisper_model, tts_model)
    # desktop.creat_desktop()

if __name__ == "__main__":
    main()
