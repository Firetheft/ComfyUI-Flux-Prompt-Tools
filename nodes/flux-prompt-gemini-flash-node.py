import os
import json
import google.generativeai as genai
from io import BytesIO
from PIL import Image
import torch
import torchaudio
from contextlib import contextmanager

p = os.path.dirname(os.path.realpath(__file__))

def get_config():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(p, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

@contextmanager
def temporary_env_var(key: str, new_value):
    old_value = os.environ.get(key)
    if new_value is not None:
        os.environ[key] = new_value
    elif key in os.environ:
        del os.environ[key]
    try:
        yield
    finally:
        if old_value is not None:
            os.environ[key] = old_value
        elif key in os.environ:
            del os.environ[key]

class FluxPromptGeminiFlashNode:

    def __init__(self, api_key=None, proxy=None):
        config = get_config()
        self.api_key = api_key or config.get("GEMINI_API_KEY")
        self.proxy = proxy or config.get("PROXY")
        if self.api_key is not None:
            self.configure_genai()

    def configure_genai(self):
        genai.configure(api_key=self.api_key, transport='rest')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "按需求自定义指令或者使用下面的预设指令！", "multiline": True}),
                "preset_prompt": (["无指令",  # 添加空值选项
                                   "文本指令-提示词增强",
                                   "文本指令-文本打标",
                                   "文本指令-翻译成中文",
                                   "文本指令-翻译成英文",
                                   "图片指令-图片描述",
                                   "图片指令-图片打标",
                                   "视频指令-视频描述",
                                   "音频指令-音频描述"],
                                  {"default": "无指令"}),  # 默认值设为空
                "input_type": (["text", "image", "video", "audio"], {"default": "text"}),
                "api_key": ("STRING", {"default": ""}),
                "proxy": ("STRING", {"default": ""})
            },
            "optional": {
                "text_input": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "audio": ("AUDIO",),
                "max_output_tokens": ("INT", {"default": 1000, "min": 1, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_content",)
    FUNCTION = "generate_content"

    CATEGORY = "📜Flux Prompt"

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

    def resize_image(self, image, max_size):
        width, height = image.size
        if width > height:
            if width > max_size:
                height = int(max_size * height / width)
                width = max_size
        else:
            if height > max_size:
                width = int(max_size * width / height)
                height = max_size
        return image.resize((width, height), Image.LANCZOS)

    def generate_content(self, prompt, preset_prompt, input_type, api_key, proxy, text_input=None, image=None, video=None, audio=None, max_output_tokens=1000, temperature=0.4):
        # 根据选定的中文选项设置 prompt
        preset_prompt_map = {
            "无指令": "",  # 空值
            "文本指令-提示词增强": "Generate a creative description",
            "文本指令-文本打标": "Transform the text into tags, separated by commas, don’t use duplicate tags.",
            "文本指令-翻译成中文": "Translate this text into Chinese",
            "文本指令-翻译成英文": "Translate this text into English",
            "图片指令-图片描述": "Describe the image in detail and accurately",
            "图片指令-图片打标": "Use tags to briefly and accurately describe this image, separated by commas, don’t use duplicate tags.",
            "视频指令-视频描述": "Describe the video in detail and accurately.",
            "音频指令-音频描述": "Describe the audio in detail and accurately."
        }

        # 获取对应的英文提示词
        prompt_from_preset = preset_prompt_map.get(preset_prompt, "")
        
        if prompt_from_preset:
            prompt = prompt_from_preset  # 将下拉菜单中的选择覆盖到 prompt 中

        config_updated = False
        if api_key and api_key != self.api_key:
            self.api_key = api_key
            config_updated = True
        if proxy != self.proxy:
            self.proxy = proxy
            config_updated = True
        
        if config_updated:
            save_config({"GEMINI_API_KEY": self.api_key, "PROXY": self.proxy})
            self.configure_genai()

        if not self.api_key:
            raise ValueError("API key is required")

        model_name = 'gemini-1.5-flash'
        model = genai.GenerativeModel(model_name)

        with temporary_env_var('HTTP_PROXY', self.proxy), temporary_env_var('HTTPS_PROXY', self.proxy):
            try:
                content = []
                if input_type == "text":
                    content = [prompt, text_input] if text_input else [prompt]
                elif input_type == "image" and image is not None:
                    pil_image = self.tensor_to_image(image)
                    pil_image = self.resize_image(pil_image, 1024)  # Resize single image to max 1024 pixels on longest side
                    content = [prompt, pil_image]
                elif input_type == "video" and video is not None:
                    if len(video.shape) == 4 and video.shape[0] > 1:  # Multiple frames
                        frame_count = video.shape[0]
                        step = max(1, frame_count // 10)  # Sample at most 10 frames
                        frames = [self.tensor_to_image(video[i]) for i in range(0, frame_count, step)]
                        frames = [self.resize_image(frame, 256) for frame in frames]  # Resize frames to 256x256
                        content = [f"This is a video with {frame_count} frames. Analyze the video content, paying attention to any changes or movements across frames:"] + frames + [prompt]
                    else:  # Single frame
                        pil_image = self.tensor_to_image(video.squeeze(0) if len(video.shape) == 4 else video)
                        pil_image = self.resize_image(pil_image, 1024)  # Treat single frame as image, resize to max 1024 pixels
                        content = ["This is a single frame from a video. Analyze the image content:", pil_image, prompt]
                elif input_type == "audio" and audio is not None:
                    waveform = audio["waveform"]
                    sample_rate = audio["sample_rate"]
                    
                    # Ensure the audio is 2D (channels, samples)
                    if waveform.dim() == 3:
                        waveform = waveform.squeeze(0)  # Remove batch dimension if present
                    elif waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)  # Add channel dimension if not present
                    
                    # Ensure the audio is mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    # Convert to 16kHz if necessary
                    if sample_rate != 16000:
                        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                    
                    # Convert to bytes
                    buffer = BytesIO()
                    torchaudio.save(buffer, waveform, 16000, format="WAV")
                    audio_bytes = buffer.getvalue()
                    
                    content = [prompt, {"mime_type": "audio/wav", "data": audio_bytes}]
                else:
                    raise ValueError(f"Invalid or missing input for {input_type}")

                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature
                )

                response = model.generate_content(content, generation_config=generation_config)
                generated_content = response.text

            except Exception as e:
                generated_content = f"Error: {str(e)}"
        
        return (generated_content,)

NODE_CLASS_MAPPINGS = {
    "FluxPromptGeminiFlash": FluxPromptGeminiFlashNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxPromptGeminiFlash": "Flux Prompt Gemini Flash",
}