import os.path
import time
ffmpeg_path = "data/ffmpeg/"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
from f5_tts.api import F5TTS
import json
import shutil
from pydub import AudioSegment
import random
import torch
import whisper
from opencc import OpenCC
import gradio as gr
import inference
import processing

cc_model = OpenCC('t2s')

def load_whisper_model(whisper_model_part):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(
        whisper_model_part,
        device=device,
        download_root=os.path.dirname("data/whisper_models")
    )
    return model

def load_f5tts_model(f5tts_model_part):
    return F5TTS(model = "F5TTS_v1_Base",ckpt_file = f5tts_model_part,vocab_file = "data/vocab.txt",ode_method="euler",use_ema=True,vocoder_local_path = None,device = None,hf_cache_dir = "data/huggingface_models")

def whisper_speech_to_text(model, audio_file_path):
    if os.path.exists(audio_file_path):
        result = model.transcribe(audio_file_path,language = "zh",temperature = 0.2,beam_size = 5)["text"].strip()
    else:
        result = None
    return cc_model.convert(result)

def get_speaker_config_json():
    folder_path = "speaker/speaker_config"
    json_files = []
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                file_name_without_ext = os.path.splitext(file)[0]
                json_files.append(file_name_without_ext)
    return json_files

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

def speaker_random_select(choice):
    return choice[random.randint(0, len(choice)-1)]

def get_voice_path():
    file_paths = []
    for root, dirs, files in os.walk("output/output_voice"):
        for file in files:
            if not file.endswith('.gitkeep'):
                file_paths.append(f"{root}/{file}")
    return file_paths

def get_pptx_path():
    file_paths = []
    for root, dirs, files in os.walk("output/output_pptx"):
        for file in files:
            if not file.endswith('.gitkeep'):
                file_paths.append(f"{root}/{file}")
    return file_paths

def get_view_path():
    file_paths = []
    for root, dirs, files in os.walk("output/output_view"):
        for file in files:
            if not file.endswith('.gitkeep'):
                file_paths.append(f"{root}/{file}")
    return file_paths

def get_time():
        timestamp = time.time()
        local_time = time.localtime(timestamp)
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S",local_time) + f".{int((timestamp - int(timestamp)) * 100):02d}"
        return time_str


def main():

    shutil.rmtree("data/temp")
    os.makedirs("data/temp")
    tts_model = load_f5tts_model("data/tts_models/pro_v2.0.pt")
    whisper_model = load_whisper_model("data/whisper_models/medium.pt")

    with gr.Blocks(theme = gr.themes.Soft()) as main_page:
        with gr.Tab(label = "人声库"):
            with gr.Column(visible = True) as speaker_edit_page:
                with gr.Row(equal_height = True):
                    def speaker_load_btn():
                        return gr.Dropdown(choices = get_speaker_config_json())
                    speaker_choose = gr.Dropdown(choices = get_speaker_config_json(), label ="选择人声", allow_custom_value = False, scale = 11)
                    speaker_load = gr.Button("加载",scale = 1,variant = "primary")
                    speaker_load.click(fn = speaker_load_btn,inputs = [],outputs = speaker_choose)
                speaker_text = gr.Textbox(label = "参考文本",interactive = False,scale = 10)
                speaker_wav = gr.Audio(type = "filepath",label = "人声文件")
                speaker_choose.change(fn = get_speaker_config_value,inputs = speaker_choose,outputs = [speaker_wav,speaker_text])
                with gr.Row():
                    def speaker_edit_btn():
                        return gr.Button(visible=False),gr.Button(visible=False),gr.Button(visible=False),gr.Button(visible = True),gr.Button(visible = True),gr.Button(visible = True),gr.Textbox(interactive = True)
                    def speaker_save_btn(file_name, edit_text):
                        file_path = f"speaker/speaker_config/{file_name}.json"
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                speaker_data = json.load(f)
                                speaker_data['reference_text'] = edit_text
                        except Exception as e:
                            print(f"Error opening speaker config file: {e}")
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(speaker_data, f, ensure_ascii=False, indent=4)
                        gr.Info("保存成功")
                        return gr.Button(visible = True),gr.Button(visible = True),gr.Button(visible = True),gr.Button(visible=False),gr.Button(visible=False),gr.Button(visible=False),gr.Textbox(interactive = False)
                    def speaker_cancel_btn():
                        return gr.Button(visible = True),gr.Button(visible = True),gr.Button(visible = True),gr.Button(visible=False),gr.Button(visible=False),gr.Button(visible=False),gr.Textbox(interactive = False)
                    def speaker_delete_btn(file_name):
                        delete_config_path = f"speaker/speaker_config/{file_name}.json"
                        delete_voice_path = f"speaker/speaker_voice/{file_name}_voice.wav"
                        os.remove(delete_config_path)
                        os.remove(delete_voice_path)
                        gr.Info("删除成功")
                        return gr.Button(visible = True),gr.Button(visible = True),gr.Button(visible = True),gr.Button(visible=False),gr.Button(visible=False),gr.Button(visible=False),gr.Textbox(interactive = False)
                    def speaker_create_btn():
                        return gr.Column(visible = False),gr.Column(visible = True)

                    random_speaker = gr.Button("随机选择", visible = True)
                    speaker_edit = gr.Button("编辑人声", visible = True)
                    speaker_create = gr.Button("上传人音",visible = True)
                    speaker_text_save = gr.Button("保存文本", visible = False,variant = "primary")
                    speaker_cancel = gr.Button("取消编辑",visible = False,variant = "secondary")
                    speaker_delete = gr.Button("删除人声",visible = False,variant = "stop")

                    random_speaker.click(fn = lambda: speaker_random_select(get_speaker_config_json()),inputs = [],outputs = speaker_choose)
                    speaker_edit.click(fn = speaker_edit_btn,inputs=[],
                        outputs = [random_speaker,speaker_edit,speaker_create,speaker_text_save,speaker_cancel,speaker_delete,speaker_text])
                    speaker_text_save.click(fn = speaker_save_btn,inputs = [speaker_choose,speaker_text],
                        outputs = [random_speaker,speaker_edit,speaker_create,speaker_text_save,speaker_cancel,speaker_delete,speaker_text])
                    speaker_cancel.click(fn = speaker_cancel_btn,inputs = [],
                        outputs = [random_speaker,speaker_edit,speaker_create,speaker_text_save,speaker_cancel,speaker_delete,speaker_text])
                    speaker_delete.click(fn = speaker_delete_btn,inputs = speaker_choose,
                        outputs = [random_speaker,speaker_edit,speaker_create,speaker_text_save,speaker_cancel,speaker_delete,speaker_text])

            with gr.Column(visible = False) as speaker_create_page:
                def get_audio_upload_text_btn(voice_temp_path):
                    return whisper_speech_to_text(whisper_model, voice_temp_path)

                create_name = gr.Textbox(label = "输入人声名称")
                audio_upload = gr.Audio(label="请上传5s~10s的单说话人音频文件", type="filepath")
                with gr.Row(equal_height = True):
                    audio_upload_text = gr.Textbox(label = "请输入人声文本",scale = 10,interactive = True,info = "输入的文本应为汉字和英文并且阿拉伯数字要转为对应的汉字和英文")
                    get_audio_upload_text = gr.Button("AI识别",scale = 2)
                    get_audio_upload_text.click(
                        fn = get_audio_upload_text_btn,
                        inputs = audio_upload,
                        outputs = audio_upload_text
                    )
                with gr.Row():
                    def create_cancel_btn():
                        return gr.Column(visible = True),gr.Column(visible = False)
                    def create_sumit_btn(file_name,voice_temp_path,voice_text):
                        try:
                            file_path = f"speaker/speaker_config/{file_name}.json"
                            file_extension = os.path.splitext(voice_temp_path)[1].lower()
                            voice_path = f"speaker/speaker_voice/{file_name}_voice.wav"
                            if file_extension != '.wav':
                                audio_tool = AudioSegment.from_file(voice_temp_path)
                                audio_tool.export(voice_path, format="wav")
                            else:
                                shutil.copy(voice_temp_path, voice_path)
                            data = {
                                "name": file_name,
                                "voice_path": voice_path,
                                "reference_text": voice_text
                            }
                            with open(file_path, 'w', encoding='utf-8') as file:
                                json.dump(data, file, indent=4, ensure_ascii=False)
                            gr.Info("上传成功")
                            return gr.Column(visible=True), gr.Column(visible=False)
                        except Exception as e:
                            gr.Error(str(e))
                            return gr.Column(visible=True), gr.Column(visible=False)
                    create_sumit = gr.Button("提交",variant = "primary")
                    create_cancel = gr.Button("取消",variant = "stop")
            speaker_create.click(
                fn = speaker_create_btn,
                inputs = [],
                outputs = [speaker_edit_page,speaker_create_page]
            )
            create_cancel.click(
                fn = create_cancel_btn,
                inputs = [],
                outputs = [speaker_edit_page,speaker_create_page]
            )
            create_sumit.click(
                fn = create_sumit_btn,
                inputs = [create_name,audio_upload,audio_upload_text],
                outputs = [speaker_edit_page,speaker_create_page]
            )
        with gr.Tab(label = "生成音频"):
            with gr.Row(equal_height = True):
                def voice_speaker_load_btn():
                    return gr.Dropdown(choices = get_speaker_config_json())
                voice_speaker_choose = gr.Dropdown(choices = get_speaker_config_json(),label = "选择人声", scale = 11,allow_custom_value = False)
                voice_speaker_load = gr.Button("加载",scale = 1,variant = "primary")
            with gr.Row(equal_height = True):
                def voice_config(ref_file, ref_text, gen_text, file_wave, target_rms, speed, cfg_strength, nfe_step,sway_sampling_coef, cross_fade_duration, fix_duration, seed, remove_silence):
                    if file_wave is None or file_wave.strip() == "":
                        timestamp = time.time()
                        local_time = time.localtime(timestamp)
                        file_wave = time.strftime("%Y-%m-%d_%H-%M-%S",local_time) + f".{int((timestamp - int(timestamp)) * 100):02d}"
                    file_wave = f"output/output_voice/{file_wave}.wav"
                    if target_rms == 0:
                        target_rms = 0.1
                    if speed == 0:
                        speed = 1
                    if cfg_strength == 0:
                        cfg_strength = 6
                    if nfe_step == 0:
                        nfe_step = 16
                    if sway_sampling_coef == 0:
                        sway_sampling_coef = -1
                    if cross_fade_duration == 0:
                        cross_fade_duration = 0.1
                    config = {
                        "ref_file": ref_file,
                        "ref_text": ref_text,
                        "gen_text": gen_text,
                        "file_wave": file_wave,
                        "target_rms": target_rms,
                        "speed": speed,
                        "cfg_strength": cfg_strength,
                        "nfe_step": nfe_step,
                        "sway_sampling_coef": sway_sampling_coef,
                        "cross_fade_duration": cross_fade_duration,
                        "fix_duration": fix_duration,
                        "seed": seed,
                        "remove_silence": remove_silence
                    }
                    voice_save_path = inference.wavs_inference(tts_model,config)
                    return voice_save_path,gr.Dropdown(value = voice_save_path,choices = get_voice_path())
                def voice_delete(delete_path):
                    if os.path.exists(delete_path):
                        os.remove(delete_path)
                    return gr.Dropdown(choices = get_voice_path())
                with gr.Column(scale = 3):
                    voice_name = gr.Textbox(label = "保存名称",interactive = True,info = "默认为当前时间")
                    voice_target_rms = gr.Number(label = "目标音量",interactive = True,info = "请输入0~1之间的浮点数,默认为0.1",value = 0.1)
                    voice_speed = gr.Number(label = "语速",interactive = True,info = "默认为1,高于1为加速低于1为减速", value = 1)
                    voice_cfg_strength = gr.Number(label = "模仿强度",interactive = True,info = "默认为6,控制语音合成时的风格或表现力强度",value = 6)
                    voice_nfe_step = gr.Number(label = "每步的采样数",interactive = True,info = "默认为16,越高越接近原音频,但生成速度会变慢",value = 16)
                    voice_sway_sampling_coef = gr.Number(label = "采样系数",interactive = True,info = "默认为-1,制音频合成中的细节",value = -1)
                    voice_cross_fade_duration = gr.Number(label = "淡入淡出时间",interactive = True,info = "默认为0.1,避免音频切换时产生突兀感",value = 0.1)
                    voice_seed = gr.Number(label = "种子",interactive = True,info = "控制生成语音的随机性,确保每次合成结果一致",value = None)
                    voice_fix_duration = gr.Number(label = "固定时长",interactive = True,info = "0为不固定时长",value = 0)
                    voice_remove_silence = gr.Checkbox(label = "去除静音",interactive = True,info = "默认为去除静音",value = True)
                with gr.Column(scale = 7):
                    voice_speaker_wav = gr.Audio(label="参考音频", type="filepath",interactive = False)
                    voice_speaker_text = gr.Textbox(label = "参考文本",interactive = False)
                    voice_speaker_input_text = gr.TextArea(label = "输入文本",interactive = True)
                    with gr.Row(equal_height = True):
                        def voice_speaker_output_load_btn():
                            return gr.Dropdown(choices = get_voice_path())
                        def voice_output_path(path):
                            return path
                        voice_speaker_output_dropdown = gr.Dropdown(label = "查看音频", choices = get_voice_path(),allow_custom_value = False,scale = 9)
                        voice_speaker_output_load = gr.Button("加载",scale = 1,variant = "primary")
                        voice_speaker_output_load.click(fn = voice_speaker_output_load_btn,inputs = [],outputs = voice_speaker_output_dropdown)
                    voice_speaker_output_wav = gr.Audio(label="生成音频", type="filepath",interactive = False)
                    with gr.Row(equal_height = True):
                        voice_speaker_sumit = gr.Button("生成",variant = "primary")
                        voice_speaker_delete = gr.Button("删除",variant = "stop")
            voice_speaker_output_dropdown.change(fn = voice_output_path,inputs = voice_speaker_output_dropdown,outputs = voice_speaker_output_wav)
            voice_speaker_sumit.click(fn = voice_config,
                inputs = [voice_speaker_wav,voice_speaker_text,voice_speaker_input_text,voice_name,voice_target_rms,voice_speed,voice_cfg_strength,voice_nfe_step,voice_sway_sampling_coef,voice_cross_fade_duration,voice_fix_duration,voice_seed,voice_remove_silence],
                outputs = [voice_speaker_output_wav,voice_speaker_output_dropdown])
            voice_speaker_delete.click(fn = voice_delete, inputs = voice_speaker_output_dropdown, outputs = voice_speaker_output_dropdown)
            voice_speaker_load.click(fn = voice_speaker_load_btn,inputs = [],outputs = voice_speaker_choose)
            voice_speaker_choose.change(fn = get_speaker_config_value,inputs = voice_speaker_choose,outputs = [voice_speaker_wav,voice_speaker_text])
        with gr.Tab(label = "PPT制作"):
            with gr.Row(equal_height = True):
                def ppt_speaker_load_btn():
                    return gr.Dropdown(choices = get_speaker_config_json())
                ppt_speaker_choose = gr.Dropdown(choices = get_speaker_config_json(),label = "选择人声", scale = 11,allow_custom_value = False)
                ppt_speaker_load = gr.Button("加载", scale=1, variant="primary")
            with gr.Row(equal_height = True):
                with gr.Column(scale = 3):
                    ppt_name = gr.Textbox(label = "保存名称",interactive = True,info = "默认为当前时间")
                    ppt_target_rms = gr.Number(label = "目标音量",interactive = True,info = "请输入0~1之间的浮点数,默认为0.1",value = 0.1)
                    ppt_speed = gr.Number(label = "语速",interactive = True,info = "默认为1,高于1为加速低于1为减速", value = 1)
                    ppt_cfg_strength = gr.Number(label = "模仿强度",interactive = True,info = "默认为6,控制语音合成时的风格或表现力强度",value = 6)
                    ppt_nfe_step = gr.Number(label = "每步的采样数",interactive = True,info = "默认为16,越高越接近原音频,但生成速度会变慢",value = 16)
                    ppt_sway_sampling_coef = gr.Number(label = "采样系数",interactive = True,info = "默认为-1,制音频合成中的细节",value = -1)
                    ppt_cross_fade_duration = gr.Number(label = "淡入淡出时间",interactive = True,info = "默认为0.1,避免音频切换时产生突兀感",value = 0.1)
                    ppt_seed = gr.Number(label = "种子",interactive = True,info = "控制生成语音的随机性,确保每次合成结果一致",value = None)
                    ppt_fix_duration = gr.Number(label = "固定时长",interactive = True,info = "0为不固定时长",value = 0)
                    ppt_remove_silence = gr.Checkbox(label = "去除静音",interactive = True,info = "默认为去除静音",value = True)
                with gr.Column(scale = 7):
                    def open_file(ppt_path):
                        absolute_path = os.path.abspath(ppt_path)
                        os.system(f"explorer.exe {absolute_path}")
                    def delete_pptx(ppt_path):
                        os.remove(ppt_path)
                        return gr.Dropdown(choices=get_pptx_path())
                    def ppt_load():
                        return gr.Dropdown(choices=get_pptx_path())
                    ppt_speaker_wav = gr.Audio(label="参考音频", type="filepath",interactive= False,scale = 2)
                    ppt_speaker_text = gr.TextArea(label = "参考文本",interactive = False,scale = 2)
                    ppt_input = gr.File(label = "上传PPT",interactive = True)
                    with gr.Row(equal_height = True):
                        ppt_output = gr.Dropdown(label = "查看ppt", choices = get_pptx_path(),allow_custom_value = False,scale = 9)
                        ppt_output_load = gr.Button("加载",scale = 1,variant = "primary")
                        ppt_output_load.click(fn = ppt_load,inputs = [],outputs = ppt_output)
                    with gr.Row(equal_height = True):
                        ppt_open = gr.Button("打开")
                        ppt_delete = gr.Button("删除",variant = "stop")
                        ppt_open.click(fn = open_file,inputs = ppt_output,outputs = [])
                        ppt_delete.click(fn = delete_pptx,inputs = ppt_output,outputs = ppt_output)
                    with gr.Tab(label = "快速制作",):
                        def ppt_config_fast(ref_file, ref_text, ppt_path, file_wave, target_rms, speed, cfg_strength, nfe_step,sway_sampling_coef, cross_fade_duration, fix_duration, seed, remove_silence):
                            if file_wave is None or file_wave.strip() == "":
                                timestamp = time.time()
                                local_time = time.localtime(timestamp)
                                file_wave = time.strftime("%Y-%m-%d_%H-%M-%S",local_time) + f".{int((timestamp - int(timestamp)) * 100):02d}"
                            if target_rms == 0:
                                target_rms = 0.1
                            if speed == 0:
                                speed = 1
                            if cfg_strength == 0:
                                cfg_strength = 6
                            if nfe_step == 0:
                                nfe_step = 16
                            if sway_sampling_coef == 0:
                                sway_sampling_coef = -1
                            if cross_fade_duration == 0:
                                cross_fade_duration = 0.1
                            config = {
                                "ref_file": ref_file,
                                "ref_text": ref_text,
                                "target_rms": target_rms,
                                "speed": speed,
                                "cfg_strength": cfg_strength,
                                "nfe_step": nfe_step,
                                "sway_sampling_coef": sway_sampling_coef,
                                "cross_fade_duration": cross_fade_duration,
                                "fix_duration": fix_duration,
                                "seed": seed,
                                "remove_silence": remove_silence
                            }
                            ppt_save_path = inference.ppt_fast_inference(tts_model,config, ppt_path, f"output/output_pptx/{file_wave}.pptx")
                            return gr.Dropdown(value=ppt_save_path, choices=get_pptx_path())
                        ppt_speaker_sumit_fast = gr.Button("生成",variant = "primary")
                        ppt_speaker_sumit_fast.click(fn = ppt_config_fast,
                                                inputs = [ppt_speaker_wav, ppt_speaker_text, ppt_input, ppt_name, ppt_target_rms, ppt_speed, ppt_cfg_strength, ppt_nfe_step, ppt_sway_sampling_coef, ppt_cross_fade_duration, ppt_fix_duration, ppt_seed, ppt_remove_silence],
                                                outputs = ppt_output)
                    with gr.Tab(label = "逐步制作",scale = 8):
                        temp_json = {}
                        def ppt_init_btn(input_ppt_path):
                            count = inference.count_textboxes(input_ppt_path)
                            for i in range(1,count + 1):
                                temp_json[f"shape_{i}"] = "NOT"
                            shutil.copy(input_ppt_path,"data/temp/ppt_temp.pptx")
                            return gr.Column(visible = False),gr.Column(visible = True),gr.Slider(minimum = 1,maximum = count,step = 1)
                        with gr.Column(visible = True) as ppt_init_page:
                            ppt_init = gr.Button("读取",variant = "primary")
                        with gr.Column(visible = False) as ppt_edit_page:
                            def ppt_slide_change(index):
                                now_ppt_text = inference.get_ppt_text(index,"data/temp/ppt_temp.pptx")
                                return now_ppt_text
                            def ppt_sumit_btn(ppt_input_name):
                                if ppt_input_name is None or ppt_input_name.strip() == "":
                                    timestamp = time.time()
                                    local_time = time.localtime(timestamp)
                                    ppt_input_name = time.strftime("%Y-%m-%d_%H-%M-%S",local_time) + f".{int((timestamp - int(timestamp)) * 100):02d}"
                                shutil.copy(inference.insert_audio_to_ppt(temp_json),f"output/output_pptx/{ppt_input_name}.pptx")
                                return gr.Column(visible = True),gr.Column(visible = False)
                            def ppt_cancel_btn():
                                return gr.Column(visible = True),gr.Column(visible = False)
                            def ppt_process_btn(target_text,ref_file, ref_text,gen_text,target_rms, speed,cfg_strength, nfe_step, sway_sampling_coef, cross_fade_duration,fix_duration, seed, remove_silence):
                                if target_rms == 0:
                                    target_rms = 0.1
                                if speed == 0:
                                    speed = 1
                                if cfg_strength == 0:
                                    cfg_strength = 6
                                if nfe_step == 0:
                                    nfe_step = 16
                                if sway_sampling_coef == 0:
                                    sway_sampling_coef = -1
                                if cross_fade_duration == 0:
                                    cross_fade_duration = 0.1
                                config = {
                                    "ref_file": ref_file,
                                    "ref_text": ref_text,
                                    "gen_text" : gen_text,
                                    "file_wave" : f"data/temp/ppt_voice_temp_{get_time()}.wav",
                                    "target_rms": target_rms,
                                    "speed": speed,
                                    "cfg_strength": cfg_strength,
                                    "nfe_step": nfe_step,
                                    "sway_sampling_coef": sway_sampling_coef,
                                    "cross_fade_duration": cross_fade_duration,
                                    "fix_duration": fix_duration,
                                    "seed": seed,
                                    "remove_silence": remove_silence
                                }
                                new_path = inference.wavs_inference(tts_model,config)
                                temp_json[f"shape_{target_text}"] = new_path
                                return new_path
                            def ppt_delete_btn(index):
                                temp_json[f"shape_{index}"] = "NOT"
                                gr.Info("已移除音频")
                                return None
                            ppt_slide = gr.Slider()
                            ppt_text_load = gr.Textbox(label = "PPT文本")
                            ppt_new_voice = gr.Audio(label = "生成音频",type="filepath",interactive= False)
                            with gr.Row(equal_height = True):
                                ppt_sumit = gr.Button("提交",variant = "primary")
                                ppt_process = gr.Button("生成音频")
                                ppt_delete = gr.Button("移除音频")
                                ppt_cancel = gr.Button("取消",variant = "stop")
                                ppt_delete.click(fn = ppt_delete_btn,inputs = ppt_slide,outputs = [])
                                ppt_sumit.click(fn = ppt_sumit_btn,inputs = ppt_name,outputs = [ppt_init_page,ppt_edit_page])
                                ppt_cancel.click(fn = ppt_cancel_btn,inputs = [],outputs = [ppt_init_page,ppt_edit_page])
                                ppt_process.click(fn = ppt_process_btn,inputs = [ppt_slide,ppt_speaker_wav, ppt_speaker_text,ppt_text_load,ppt_target_rms,ppt_speed,ppt_cfg_strength,ppt_nfe_step,ppt_sway_sampling_coef,ppt_cross_fade_duration,ppt_fix_duration,ppt_seed,ppt_remove_silence],outputs = ppt_new_voice)
                            ppt_slide.change(fn = ppt_slide_change,inputs = ppt_slide,outputs = ppt_text_load)
                        ppt_init.click(fn = ppt_init_btn,inputs = ppt_input,outputs = [ppt_init_page,ppt_edit_page,ppt_slide])
            ppt_speaker_load.click(fn = ppt_speaker_load_btn,inputs = [],outputs = ppt_speaker_choose)
            ppt_speaker_choose.change(fn = get_speaker_config_value,inputs = ppt_speaker_choose,outputs = [ppt_speaker_wav,ppt_speaker_text])
        with gr.Tab(label = "配音替换"):
            with gr.Row(equal_height = True):
                def view_speaker_load_btn():
                    return gr.Dropdown(choices=get_speaker_config_json())
                view_speaker_choose = gr.Dropdown(label = "选择人声",choices = get_speaker_config_json(),allow_custom_value = False,scale = 11)
                view_speaker_load = gr.Button("加载",variant = "primary")
                view_speaker_load.click(fn = view_speaker_load_btn,inputs = [],outputs = view_speaker_choose)
            with gr.Row(equal_height = True):
                with gr.Column(scale = 3):
                    view_name = gr.Textbox(label="保存名称", interactive=True, info="默认为当前时间")
                    view_target_rms = gr.Number(label="目标音量", interactive=True,info="请输入0~1之间的浮点数,默认为0.1", value=0.1)
                    view_speed = gr.Number(label="语速", interactive=True, info="默认为1,高于1为加速低于1为减速",value=1)
                    view_cfg_strength = gr.Number(label="模仿强度", interactive=True,info="默认为6,控制语音合成时的风格或表现力强度", value=6)
                    view_nfe_step = gr.Number(label="每步的采样数", interactive=True,info="默认为16,越高越接近原音频,但生成速度会变慢", value=16)
                    view_sway_sampling_coef = gr.Number(label="采样系数", interactive=True,info="默认为-1,制音频合成中的细节", value=-1)
                    view_cross_fade_duration = gr.Number(label="淡入淡出时间", interactive=True,info="默认为0.1,避免音频切换时产生突兀感", value=0.1)
                    view_seed = gr.Number(label="种子", interactive=True,info="控制生成语音的随机性,确保每次合成结果一致", value=None)
                    view_fix_duration = gr.Number(label="固定时长", interactive=True, info="0为不固定时长", value=0)
                    view_remove_silence = gr.Checkbox(label="去除静音", interactive=True, info="默认为去除静音",value=True)
                    view_caption = gr.Checkbox(label = "加入字幕",interactive=True, info="默认为加入字幕",value=True)
                with gr.Column(scale = 7):
                    view_config = {}
                    text_time_config = {}
                    def view_input_btn(input_view_path,ref_file, ref_text, target_rms, speed, cfg_strength,nfe_step, sway_sampling_coef, cross_fade_duration, fix_duration, seed,remove_silence):
                        gr.Info("正在加载视频请等待")
                        if os.path.exists("data/temp/view_temp"):
                            shutil.rmtree("data/temp/view_temp")
                        os.makedirs("data/temp/view_temp")
                        if target_rms == 0:
                            target_rms = 0.1
                        if speed == 0:
                            speed = 1
                        if cfg_strength == 0:
                            cfg_strength = 6
                        if nfe_step == 0:
                            nfe_step = 16
                        if sway_sampling_coef == 0:
                            sway_sampling_coef = -1
                        if cross_fade_duration == 0:
                            cross_fade_duration = 0.1
                        input_view_config = {
                            "ref_file": ref_file,
                            "ref_text": ref_text,
                            "target_rms": target_rms,
                            "speed": speed,
                            "cfg_strength": cfg_strength,
                            "nfe_step": nfe_step,
                            "sway_sampling_coef": sway_sampling_coef,
                            "cross_fade_duration": cross_fade_duration,
                            "fix_duration": fix_duration,
                            "seed": seed,
                            "remove_silence": remove_silence
                        }
                        if not os.path.exists("data/temp/view_temp"):
                            os.makedirs("data/temp/view_temp")
                        if not os.path.exists("data/temp/view_temp/clips"):
                            os.makedirs("data/temp/view_temp/clips")
                        if not os.path.exists("data/temp/view_temp/new_clips"):
                            os.makedirs("data/temp/view_temp/new_clips")
                        if not os.path.exists("data/temp/view_temp/separate"):
                            os.makedirs("data/temp/view_temp/separate")
                        vocal_path,bass_path,drums_path,other_path = processing.separate_audio(input_view_path,"data/temp/view_temp/separate")
                        processing.combine_accompaniments(bass_path,drums_path,other_path,"data/temp/view_temp/accompaniment.wav")
                        processing.split_audio_on_silence(vocal_path,"data/temp/view_temp/clips")
                        new_whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
                        new_whisper_model = whisper.load_model(
                            "data/whisper_models/base.pt",
                            device=new_whisper_device,
                            download_root=os.path.dirname("data/whisper_models")
                        )
                        new_text_time_config = processing.recognize_audio_files(new_whisper_model)
                        view_config.update(input_view_config)
                        text_time_config.update(new_text_time_config)
                        processing.traverse_audio_dict(text_time_config)
                        gr.Info("视频加载完成 请加载进度条")
                        return gr.Column(visible = False),gr.Column(visible = True)
                    def get_view_slider_value():
                        return len(text_time_config)
                    def view_slider_change(index):
                        return text_time_config[f"output_clip_{index}.wav"].text,f"data/temp/view_temp/clips/output_clip_{index}.wav"
                    def view_create_btn(caption,intput_view_path,file_wave):
                        if file_wave is None or file_wave.strip() == "":
                            timestamp = time.time()
                            local_time = time.localtime(timestamp)
                            file_wave = time.strftime("%Y-%m-%d_%H-%M-%S",local_time) + f".{int((timestamp - int(timestamp)) * 100):02d}"
                        output_path = f"output/output_view/{file_wave}.mp4"
                        processing.mix_audio_with_accompaniment(text_time_config)
                        if caption:
                            view_path = processing.make_view(intput_view_path)
                            processing.add_subtitles_to_video(text_time_config,view_path,output_path)
                        else:
                            processing.make_view_save(intput_view_path,output_path)
                        return gr.Column(visible = True),gr.Column(visible = False),gr.Dropdown(choices = get_view_path())
                    def view_process_btn(index,gen_text):
                        new_view_config = view_config.copy()
                        new_view_config["gen_text"] = gen_text
                        file_wave = f"data/temp/view_temp/new_clips/new_output_clip_{index}.wav"
                        new_view_config["file_wave"] = file_wave
                        text_time_config[f"output_clip_{index}.wav"].wav_path = file_wave
                        text_time_config[f"output_clip_{index}.wav"].text = gen_text
                        new_clip_path = inference.wavs_inference(tts_model,new_view_config)
                        return new_clip_path
                    def view_delete_btn(index):
                        text_time_config[f"output_clip_{index}.wav"].wav_path = "NOT"
                        gr.Info(f"片段{index}的人声删除完成")
                    def view_cancel_btn():
                        return gr.Column(visible = True),gr.Column(visible = False)
                    view_speaker_wav = gr.Audio(label = "参考音频",type="filepath",interactive= False)
                    view_speaker_text = gr.Textbox(label = "参考文本",interactive= False)
                    view = gr.Video(label = " 上传视频")
                    with gr.Column(visible = True) as view_input_column:
                        view_input = gr.Button("开始制作",variant = "primary")
                        with gr.Row(equal_height = True):
                            def load_view(view_path):
                                return view_path
                            def view_load_btn():
                                return gr.Dropdown(choices = get_view_path())
                            def view_output_delete_btn(path):
                                os.remove(path)
                                return gr.Dropdown(choices = get_view_path())
                            view_dropdown = gr.Dropdown(label = "选择视频",choices = get_view_path(),scale = 8)
                            view_load = gr.Button("加载",variant = "primary",scale = 2)
                            view_output_delete = gr.Button("删除",variant = "stop",scale = 2)
                            view_output_delete.click(fn = view_output_delete_btn,inputs = view_dropdown,outputs = view_dropdown)
                            view_load.click(fn = view_load_btn,inputs = [],outputs = view_dropdown)
                        view_output = gr.Video(label = "生成视频",interactive= False)
                        view_dropdown.change(fn = load_view,inputs = view_dropdown,outputs = view_output)
                    with gr.Column(visible = False) as view_output_column:
                        with gr.Row(equal_height = True):
                            def view_clips_load_btn():
                                return gr.Slider(label="进度条",minimum=1,maximum=get_view_slider_value(),step=1)
                            view_slider = gr.Slider(label="进度条",minimum=1,maximum=get_view_slider_value(),step=1,scale = 8)
                            view_clips_load = gr.Button("加载进度条",variant = "primary",scale = 2)
                            view_clips_load.click(fn = view_clips_load_btn,inputs = [],outputs = view_slider)
                        view_wav_input = gr.Audio(label = "视频音频",type = "filepath",interactive= False)
                        view_wav_output = gr.Audio(label = "片段音频",interactive= False)
                        view_text = gr.Textbox(label = "音频文本")
                        with gr.Row():
                            view_create = gr.Button("开始替换",variant = "primary")
                            view_process = gr.Button("生成配音")
                            view_delete = gr.Button("删除配音")
                            view_cancel = gr.Button("取消",variant = "stop")
                    view_slider.change(fn = view_slider_change, inputs = view_slider,outputs = [view_text,view_wav_input])
                    view_input.click(fn = view_input_btn,inputs = [view,view_speaker_wav,view_speaker_text,view_target_rms,view_speed,view_cfg_strength,view_nfe_step,view_sway_sampling_coef,view_cross_fade_duration,view_fix_duration,view_seed,view_remove_silence],outputs = [view_input_column,view_output_column])
                    view_create.click(fn = view_create_btn,inputs = [view_caption,view,view_name],outputs = [view_input_column,view_output_column,view_dropdown])
                    view_process.click(fn = view_process_btn,inputs = [view_slider,view_text],outputs = view_wav_output)
                    view_delete.click(fn = view_delete_btn,inputs = view_slider,outputs = None)
                    view_cancel.click(fn = view_cancel_btn,inputs = [],outputs = [view_input_column,view_output_column])
                view_speaker_choose.change(fn = get_speaker_config_value,inputs = view_speaker_choose,outputs = [view_speaker_wav,view_speaker_text])
    main_page.launch()

if __name__ == "__main__":
    main()
