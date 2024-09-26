import argparse
import os
import yaml
import torch
import numpy as np
from librosa import load
from librosa.feature import melspectrogram

import os
import numpy as np

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.preprocessed_dir = config["path"]["preprocessed_path"]  # 전처리된 데이터 저장 경로 추가
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.n_mel_channels = config["preprocessing"]["mel"]["n_mel_channels"]
        self.mel_fmin = config["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = config["preprocessing"]["mel"]["mel_fmax"]

    def process_audio(self, wav_path):
        # Load audio file
        wav, _ = load(wav_path, sr=self.sampling_rate)
        print(f"Loaded {wav_path} for mel-spectrogram processing.")
        
        # Generate mel spectrogram
        mel_spectrogram = melspectrogram(
            y=wav,
            sr=self.sampling_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax
        )
        print(f"Mel spectrogram generated for {wav_path}.")
        
        # Compute energy (sum of the mel spectrogram)
        energy = np.sum(mel_spectrogram, axis=0)
        print(f"Energy calculated for {wav_path}.")
        
        # 멜 스펙트로그램과 에너지를 파일로 저장
        mel_save_path = os.path.join(self.preprocessed_dir, "mel", f"{os.path.basename(wav_path).split('.')[0]}.npy")
        energy_save_path = os.path.join(self.preprocessed_dir, "energy", f"{os.path.basename(wav_path).split('.')[0]}.npy")
        
        # 저장할 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(mel_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(energy_save_path), exist_ok=True)

        # 멜 스펙트로그램과 에너지 저장
        np.save(mel_save_path, mel_spectrogram)
        np.save(energy_save_path, energy)
        print(f"Mel spectrogram and energy saved for {wav_path}")
        
        return mel_spectrogram, energy


    def build_from_path(self):
        # filelists 내의 파일들을 직접 가져옴
        with open(self.config["data"]["training_files"], "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            wav_path, transcript = line.strip().split("|")
            wav_path = os.path.join(self.in_dir, wav_path)

            if os.path.exists(wav_path):
                # 음성 파일 처리 중 로그 출력
                print(f"Processing {wav_path} with transcript: {transcript}")
                # 멜 스펙트로그램 생성
                mel_output, energy = self.process_audio(wav_path)
                
                # 추가적으로 전처리한 데이터를 저장하는 코드도 로그로 출력
                print(f"Saving mel spectrogram for {wav_path}")
            else:
                print(f"File {wav_path} not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    # YAML 파일 로드
    config = yaml.load(open(args.config, "r", encoding="utf-8"), Loader=yaml.FullLoader)

    # 전처리 실행
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
