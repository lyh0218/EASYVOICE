import logging
import os
from typing import Dict
from pydub import AudioSegment
from pydub.silence import split_on_silence
import subprocess
import sys
from opencc import OpenCC

'''
    多功能音频/视频处理工具
'''


# 文本时间类
class TextTime:
    def __init__(self, text, time):
        self.text = text
        self.time = time
        self.wav_path = "NOT"

# mp4 -> wav
def mp4_to_wav_pydub(mp4_path, wav_path):
    audio = AudioSegment.from_file(mp4_path, format="mp4")
    audio = audio.set_frame_rate(22050).set_channels(1) #采样率 单声道
    audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])
    return wav_path
# 人声分离
def separate_audio(input_audio_file, output_dir):
    input_audio_file = mp4_to_wav_pydub(input_audio_file,"data/temp/view_temp/input.wav")
    if os.name == "nt":
        possible_paths = [
            os.path.join(sys.prefix, "Scripts", "demucs.exe"),
            os.path.join(os.environ.get("USERPROFILE", ""), ".local", "bin", "demucs.exe")
        ]
    else:
        possible_paths = [
            os.path.join(sys.prefix, "bin", "demucs"),
            os.path.join(os.environ.get("HOME", ""), ".local", "bin", "demucs")
        ]
    demucs_path = None
    for path in possible_paths:
        if os.path.exists(path):
            demucs_path = path
            break
    if demucs_path is None:
        print("❌ 无法找到 demucs 可执行文件，请确保已安装 demucs。")
        return
    if not os.path.exists(input_audio_file):
        print(f"❌ 输入文件不存在: {input_audio_file}")
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📢 开始分离音频: {input_audio_file}")
        command = [
            demucs_path,
            '-n', 'htdemucs',
            input_audio_file,
            '-o', output_dir
        ]
        subprocess.run(command, check=True)
        model_name = 'htdemucs'
        input_name = os.path.splitext(os.path.basename(input_audio_file))[0]
        vocal_path = os.path.join(output_dir, model_name, input_name, 'vocals.wav')
        bass_path = os.path.join(output_dir, model_name, input_name, 'bass.wav')
        drums_path = os.path.join(output_dir, model_name, input_name, 'drums.wav')
        other_path = os.path.join(output_dir, model_name, input_name, 'other.wav')
        accompaniment_path = os.path.join(output_dir, model_name, input_name, 'accompaniment.wav')
        if os.path.exists(vocal_path):
            print(f"✅ 人声分离完成：{vocal_path}")
        else:
            print("⚠️ 未找到人声分离结果，请检查输出目录。")
        if os.path.exists(accompaniment_path):
            print(f"✅ 伴奏分离完成：{accompaniment_path}")
        else:
            print("⚠️ 未找到伴奏分离结果，请检查输出目录。")
        return [vocal_path,bass_path,drums_path,other_path]
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

def split_audio_on_silence(vocal_output_path, output_folder):
    """按静音进行分段并保存"""
    try:
        # 加载分离后的人声音频
        audio = AudioSegment.from_wav(vocal_output_path)
        # 按静音进行分段
        chunks = split_on_silence(audio, min_silence_len=100, silence_thresh=-45, keep_silence=200, seek_step=10)
        # 保存分段音频
        os.makedirs(output_folder, exist_ok=True)
        valid_count = 1
        for i, chunk in enumerate(chunks):
            # 调整音频以符合 训练要求
            chunk = chunk.set_frame_rate(22050).set_sample_width(2).set_channels(1)
            # 保存音频并使用有序编号
            output_path = os.path.join(output_folder, f"output_clip_{valid_count}.wav")
            chunk.export(output_path, format="wav")
            valid_count += 1
        logging.info(f"音频分段保存完成，共保存 {valid_count - 1} 个片段")
    except Exception as e:
        logging.error(f"音频分段保存时出错: {e}")

def get_wav_text(model,audio_file_path):
    cc_model = OpenCC('t2s')
    if os.path.exists(audio_file_path):
        result = model.transcribe(audio_file_path, language="zh", temperature=0.2, beam_size=5)["text"].strip()
        cc_model.convert(result)
    else:
        result = None
    return result

def recognize_audio_files(model):
    import wave
    audio_folder = "data/temp/view_temp/clips"
    result_dict = {}
    total_time = 0
    def get_file_number(file_name):
        try:
            return int(file_name.split("_")[-1].split(".")[0])
        except (IndexError, ValueError):
            return float('inf')
    audio_files = sorted(os.listdir(audio_folder), key=get_file_number)
    for filename in audio_files:
        if filename.startswith("output_clip_") and filename.endswith(".wav"):
            audio_path = os.path.join(audio_folder, filename)
            try:
                with wave.open(audio_path, 'r') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    audio_duration = frames / float(rate)
                text = get_wav_text(model, audio_path)
                result_dict[filename] = TextTime(text, total_time)
                total_time += audio_duration * 1000
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")
    return result_dict

def traverse_audio_dict(audio_dict):
    for audio_file, text_time_obj in audio_dict.items():
        print(f"音频文件名: {audio_file}")
        print(f"识别文本: {text_time_obj.text}")
        print(f"时间戳: {text_time_obj.time}")
        print("-" * 30)

def combine_accompaniments(bass_path, drums_path, other_path, output_path):
    bass = AudioSegment.from_wav(bass_path)
    drums = AudioSegment.from_wav(drums_path)
    other = AudioSegment.from_wav(other_path)

    combined = bass.overlay(drums).overlay(other)
    combined.export(output_path, format="wav")
    return output_path

def mix_audio_with_accompaniment(clip_dict):
    accompaniment_path = "data/temp/view_temp/accompaniment.wav"
    try:
        accompaniment = AudioSegment.from_wav(accompaniment_path)
        for clip_name, clip_obj in clip_dict.items():
            if clip_obj.wav_path != "NOT":
                # 加载要插入的音频
                audio_clip = AudioSegment.from_wav(clip_obj.wav_path)
                # 在指定时间插入音频
                insert_time = clip_obj.time
                if insert_time <= len(accompaniment):
                    accompaniment = accompaniment.overlay(audio_clip, position=insert_time)
                else:
                    print(f"插入时间 {insert_time} 超出伴奏长度，跳过 {clip_name}。")
        output_path = "data/temp/view_temp/output.wav"
        accompaniment.export(output_path, format="wav")
        print(f"混合后的音频已保存到 {output_path}")
    except FileNotFoundError:
        print("错误：未找到伴奏文件或音频剪辑文件。")
    except Exception as e:
        print(f"发生未知错误：{e}")

def make_view(view_path):
    audio_path = "data/temp/view_temp/output.wav"
    output_path = "data/temp/view_temp/output_new.mp4"
    command = [
        "ffmpeg",
        "-i", view_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    subprocess.run(command)
    return output_path

def make_view_save(view_path,output_path):
    audio_path = "data/temp/view_temp/output.wav"
    command = [
        "ffmpeg",
        "-i", view_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    subprocess.run(command)
    return output_path

def add_subtitles_to_video(audio_video_dict: Dict[str, TextTime], video_path: str, output_video_path: str):
    """
    使用字典中的音频信息为视频添加字幕。
    :param audio_video_dict: 字典格式 {音频文件路径: TextTime对象}
    :param video_path: 视频文件路径
    :param output_video_path: 输出视频文件路径
    """
    subtitle_file = "subtitles.srt"

    # 创建 SRT 字幕文件
    with open(subtitle_file, "w", encoding="utf-8") as srt_file:
        idx = 1
        for audio_file, text_time in audio_video_dict.items():
            # 获取字典中的 TextTime 对象
            text = text_time.text
            start_time_ms = text_time.time  # 字幕的开始时间，单位为毫秒
            wav_path = text_time.wav_path

            # 如果 wav_path 为 "NOT"，跳过此条记录
            if wav_path == "NOT":
                continue

            # 获取音频文件时长
            audio_file_path = os.path.abspath(wav_path)  # 获取音频文件的绝对路径
            if not os.path.exists(audio_file_path):
                print(f"Error: Audio file '{wav_path}' not found.")
                continue

            audio = AudioSegment.from_wav(audio_file_path)
            audio_duration_ms = len(audio)  # 获取音频文件的时长，单位是毫秒

            # 计算结束时间
            start_time_sec = start_time_ms / 1000  # 将开始时间转换为秒
            end_time_sec = (start_time_ms + audio_duration_ms) / 1000  # 结束时间 = 开始时间 + 音频时长（秒）

            # 写入 SRT 格式字幕内容
            srt_file.write(f"{idx}\n")
            srt_file.write(f"{format_time(start_time_sec)} --> {format_time(end_time_sec)}\n")
            srt_file.write(f"{text}\n\n")

            idx += 1

    # 使用 subprocess 执行 ffmpeg 命令行，添加字幕
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles={subtitle_file}",
        "-c:a", "copy",  # 保留原音频流不变
        output_video_path
    ]

    subprocess.run(command, check=True)  # 执行命令行

    # 删除临时 SRT 文件
    os.remove(subtitle_file)


def format_time(seconds: float) -> str:
    """将秒转换为 SRT 时间格式 (HH:MM:SS,MS)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"
