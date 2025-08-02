import torch
from cosyvoice.utils.file_utils import load_wav
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import sys
import os
# sys.path.append('third_party/Matcha-TTS')  # 导入Matcha-TTS的路径到模块搜索路径
# from matcha.utils.audio import mel_spectrogram
import whisper
from rich import print as rprint
import torch.nn as nn
import time
import onnxruntime as ort
import numpy as np

# 为减少导出库 暴露函数
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec



# 将tensor保存为语言无关的torchScript文件
class DictToScript(nn.Module):
    def __init__(self, data_dict):
        super(DictToScript, self).__init__()
        # 遍历字典，将每个子键的张量注册为缓冲区
        for key, sub_dict in data_dict.items():
            for sub_key, tensor in sub_dict.items():
                # 注册缓冲区，命名格式为 {key}_{sub_key}
                self.register_buffer(f"{key}_{sub_key}", tensor)

def main():
    if len(sys.argv) < 2:
        rprint("[red]请按顺序同时指定音色id和克隆音频文件的路径 例: 中文女 ./dir/中文女.wav [/red]")
        return 0
    elif len(sys.argv) < 3:
        rprint("[red]请按顺序同时指定音色id和克隆音频文件的路径 例: 中文女 ./dir/中文女.wav[/red]")
        return 0
    else:
        spkId = sys.argv[1]
        path = sys.argv[2]
        rprint(f"[green]音色id: {spkId}[/green]")
        rprint(f"[green]克隆音频文件路径：{path}[/green]")
        if not os.path.exists(path):
            rprint(f"[red]克隆音频文件{path}不存在[/red]")
            return 0
    
    device = torch.device('cpu')
    if os.path.exists(f"./spkInfo/spkInfo.pth"):
        spkInfo = torch.load(f"./spkInfo/spkInfo.pth", map_location=device)
    else:
        if not os.path.exists("./spkInfo"):
            os.mkdir("./spkInfo")
        spkInfo = {}
    
    # 1 获取embedding
    sample_rate = 16000
    prompt_speech_16k = load_wav(path, sample_rate)

    start = time.time()
    # 1 获取embedding
    feat = kaldi.fbank(prompt_speech_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    campplus_session = ort.InferenceSession("pretrained_models\CosyVoice2-0.5B\campplus.onnx")
    embedding = campplus_session.run(None, {campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    embedding = torch.tensor([embedding]).to(device)

    # 2 获取speecn_feat
    prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(prompt_speech_16k)
    speech_feat = mel_spectrogram(prompt_speech_resample, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0, fmax=8000, center=False).squeeze(dim=0).transpose(0, 1).to(device)
    speech_feat = speech_feat.unsqueeze(dim=0)

    # 3 获取speech_token
    assert prompt_speech_16k.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
    feat = whisper.log_mel_spectrogram(prompt_speech_16k, n_mels=128)
    speech_tokenizer_session = ort.InferenceSession("pretrained_models\CosyVoice2-0.5B\speech_tokenizer_v2.onnx")
    speech_token = speech_tokenizer_session.run(None, 
                                                {speech_tokenizer_session.get_inputs()[0].name: 
                                                feat.detach().cpu().numpy(), 
                                                speech_tokenizer_session.get_inputs()[1].name: 
                                                np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
    speech_token = torch.tensor([speech_token], dtype=torch.int32).to(device)

    rprint(f'[green]训练音色[{spkId}]完成, 耗时{time.time()-start:.2f}秒[/green]')

    # 4 保存spkInfo
    spkInfo[spkId] = {'embedding': embedding, 'speech_feat': speech_feat, 'speech_token': speech_token}

    # 创建模块实例
    wrapper = DictToScript(spkInfo)
    # 转换为 TorchScript 模块
    script_module = torch.jit.script(wrapper)
    # 保存为 TorchScript 模型
    script_module.save("./spkInfo/spkInfo.jit")
    # 保存为 torch python 模型
    torch.save(spkInfo, "./spkInfo/spkInfo.pth")

    rprint("[green]保存spkInfo成功[/green]")

if __name__ == "__main__":
    main()