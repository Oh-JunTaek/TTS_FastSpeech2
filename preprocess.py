import argparse
import os
import yaml
import torch
import numpy as np
from librosa import load
from librosa.feature import melspectrogram
import random  # 파일 분할을 위해 추가
import json
import pyworld as pw

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.preprocessed_dir = config["path"]["preprocessed_path"]  # 전처리된 데이터 저장 경로 추가
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.n_mel_channels = config["preprocessing"]["mel"]["n_mel_channels"]
        self.mel_fmin = config["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = config["preprocessing"]["mel"]["mel_fmax"]
        self.val_size = config["preprocessing"]["val_size"]  # 검증 데이터 비율 추가

    def process_audio(self, wav_path):
        # Load audio file
        wav, _ = load(wav_path, sr=self.sampling_rate)
        print(f"Loaded {wav_path} for mel-spectrogram processing.")
        
        # 피치 추출 추가 (pyworld 사용)
        pitch, _ = pw.dio(wav.astype(np.float64), self.sampling_rate, frame_period=1000 * 256 / self.sampling_rate)
        pitch = pw.stonemask(wav.astype(np.float64), pitch, _, self.sampling_rate)
        print(f"Pitch extracted for {wav_path}.")
        
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
        
        # 멜 스펙트로그램, 에너지, 피치 데이터를 저장할 경로 지정
        mel_save_path = os.path.join(self.preprocessed_dir, "mel", f"{os.path.basename(wav_path).split('.')[0]}.npy")
        energy_save_path = os.path.join(self.preprocessed_dir, "energy", f"{os.path.basename(wav_path).split('.')[0]}.npy")
        pitch_save_path = os.path.join(self.preprocessed_dir, "pitch", f"{os.path.basename(wav_path).split('.')[0]}.npy")
        
        # 저장할 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(mel_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(energy_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(pitch_save_path), exist_ok=True)

        # 멜 스펙트로그램, 에너지, 피치 저장
        np.save(mel_save_path, mel_spectrogram)
        np.save(energy_save_path, energy)
        np.save(pitch_save_path, pitch)
        print(f"Mel spectrogram, energy, and pitch saved for {wav_path}")
        
        return mel_spectrogram, energy, pitch

    def split_data(self, filelist, val_size):
        """훈련 데이터와 검증 데이터를 분리합니다."""
        random.shuffle(filelist)
        val_count = int(len(filelist) * (val_size / 100))
        train_data = filelist[val_count:]
        val_data = filelist[:val_count]
        return train_data, val_data

    def save_to_file(self, data, filename):
        """훈련 데이터나 검증 데이터를 파일로 저장합니다."""
        with open(filename, "w", encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")

    def build_from_path(self):
        all_pitches = []
        all_energies = []

        # filelists 내의 파일들을 직접 가져옴
        with open(self.config["data"]["training_files"], "r", encoding="utf-8") as f:
            lines = f.readlines()

        processed_lines = []
        for line in lines:
            wav_path, transcript = line.strip().split("|")
            wav_path = os.path.join(self.in_dir, wav_path)

            if os.path.exists(wav_path):
                print(f"Processing {wav_path} with transcript: {transcript}")
                mel_output, energy, pitch = self.process_audio(wav_path)
                
                all_pitches.extend(pitch)
                all_energies.extend(energy)
                # 처리된 데이터 저장
                speaker_id = "speaker_1"
                processed_lines.append(f"{wav_path}|{speaker_id}|{transcript}|{transcript}")
            else:
                print(f"File {wav_path} not found")

        # 피치와 에너지의 평균 및 표준편차 계산
        stats = {
            "pitch": {
                "mean": float(np.mean(all_pitches)) if all_pitches else 0.0,  # float()으로 변환
                "std": float(np.std(all_pitches)) if all_pitches else 0.0,   # float()으로 변환
            },
            "energy": {
                "mean": float(np.mean(all_energies)),  # float()으로 변환
                "std": float(np.std(all_energies)),   # float()으로 변환
            },
        }

        # stats.json 파일로 저장
        with open(os.path.join(self.preprocessed_dir, "stats.json"), "w") as f:
            json.dump(stats, f)
        print("Stats.json file created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    # YAML 파일 로드
    config = yaml.load(open(args.config, "r", encoding="utf-8"), Loader=yaml.FullLoader)

    # 전처리 실행
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()