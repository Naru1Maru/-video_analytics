"""
video_pipeline_ucf101.py

Инженерный видеопайплайн для задач Семинара 2, адаптированный под датасет UCF101.
Содержит:
- загрузку UCF101 через kagglehub;
- PyAV-декодер клипов (CFR/VFR, PTS/time_base);
- Dataset/DataLoader для видеоклипов;
- измерение throughput, latency, jitter;
- профилирование с torch.profiler;
- prefetch/pinned memory;
- overlap CPU/GPU (очередь + CUDA Streams);
- сравнение PyAV и decord (с fallback CPU, если нет CUDA);
- GPU-препроцессинг через torchvision.transforms.v2;
- near-real-time пайплайн decode → preprocess → infer;
- компактную 3D CNN-модель SmallC3D под UCF101 (101 класс).

Каждая "задача" семинара вынесена в отдельную функцию run_taskX_*, которую можно
вызывать по очереди.
"""

from __future__ import annotations

import time
import threading
import queue
import random 
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity, record_function

# PyAV – основной декодер
import av

# torchvision v2 (GPU-friendly transforms)
try:
    import torchvision.transforms.v2 as v2
    from torchvision.transforms.v2 import functional as F_v2  # noqa: F401
except Exception:
    v2 = None
    F_v2 = None

# decord для аппаратного декодирования
try:
    from decord import VideoReader, cpu as decord_cpu, gpu as decord_gpu
    from decord._ffi.base import DECORDError
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False
    DECORDError = Exception  # заглушка

# kagglehub для загрузки UCF101
try:
    import kagglehub
    HAS_KAGGLEHUB = True
except Exception:
    HAS_KAGGLEHUB = False


def set_global_seed(seed: int = 42) -> None:
    """
    Фиксирует сиды для Python, NumPy и PyTorch.
    Не гарантирует абсолютно одинаковое время работы,
    но убирает стохастику из модели/датасета.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Для максимальной детерминированности сворачиваем авто-тюнинг cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
# =====================================================
#  Утилита: получить список видео из UCF101 (KaggleHub)
# =====================================================

def get_ucf101_videos_kaggle(limit_videos: int = 32) -> List[str]:
    """
    Скачивает (или берёт из кэша) датасет UCF101 с Kaggle через kagglehub
    и возвращает список путей к видеофайлам.

    Параметры
    ---------
    limit_videos : int
        Ограничить число возвращаемых видео (для быстрых экспериментов).

    Возвращает
    ----------
    List[str] : список путей к видеофайлам.
    """
    if not HAS_KAGGLEHUB:
        raise ImportError(
            "Модуль kagglehub не найден. "
            "Установите его через `pip install kagglehub` "
            "или передавайте свои пути к видео напрямую."
        )

    dataset_path = kagglehub.dataset_download("matthewjansen/ucf101-action-recognition")
    root = Path(dataset_path)
    video_exts = {".avi", ".mp4", ".mkv", ".webm"}

    all_videos = sorted(
        str(p)
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in video_exts
    )
    if not all_videos:
        raise RuntimeError(
            f"В каталоге {dataset_path} не найдено видеофайлов. "
            "Проверьте структуру датасета UCF101."
        )

    if limit_videos is not None and limit_videos > 0:
        return all_videos[:limit_videos]
    return all_videos

# %%
# ============================
#  ЗАДАЧА 1: базовый декодер
# ============================

def read_clip(
    filename: str,
    start_sec: float = 0.0,
    num_frames: int = 16,
    stride: int = 2,
    return_info: bool = False,
) -> np.ndarray | Tuple[np.ndarray, dict]:
    """
    Декодирует клип из видео с помощью PyAV.

    Параметры
    ---------
    filename : str
        Путь к видеофайлу.
    start_sec : float
        Время начала клипа в секундах (по PTS * time_base).
    num_frames : int
        Число кадров в клипе (T).
    stride : int
        Шаг по кадрам: берём каждый `stride`-й кадр после start_sec.
    return_info : bool
        Если True – вернуть также словарь с индексами кадров и оценкой FPS.

    Возвращает
    ----------
    clip : np.ndarray
        Массив формы (T, H, W, 3), dtype=uint8 (меньше или равно num_frames).
    info : dict (опционально)
        {
            "frame_indices": List[int],
            "timestamps":   List[float],
            "fps_est":      float
        }
    """
    if stride <= 0:
        raise ValueError("stride должен быть > 0")

    container = av.open(filename)
    stream = container.streams.video[0]

    time_base = float(stream.time_base)
    start_pts = int(start_sec / time_base)

    frames_out: List[np.ndarray] = []
    frame_indices: List[int] = []
    timestamps: List[float] = []

    frame_idx = 0
    first_selected_idx: Optional[int] = None

    for frame in container.decode(video=0):
        if frame.pts is None:
            frame_idx += 1
            continue

        pts = frame.pts
        t_sec = pts * time_base

        if pts < start_pts:
            frame_idx += 1
            continue

        if first_selected_idx is None:
            first_selected_idx = frame_idx

        if (frame_idx - first_selected_idx) % stride == 0:
            rgb = frame.to_rgb().to_ndarray()
            frames_out.append(rgb)
            frame_indices.append(frame_idx)
            timestamps.append(t_sec)

            if len(frames_out) >= num_frames:
                break

        frame_idx += 1

    container.close()

    if not frames_out:
        raise RuntimeError(
            f"Не удалось извлечь ни одного кадра из {filename} "
            f"(start_sec={start_sec})"
        )

    clip = np.stack(frames_out, axis=0)

    info = {}
    if len(timestamps) > 1:
        dur = timestamps[-1] - timestamps[0]
        fps_est = (len(timestamps) - 1) / dur if dur > 0 else 0.0
    else:
        fps_est = 0.0

    info["frame_indices"] = frame_indices
    info["timestamps"] = timestamps
    info["fps_est"] = fps_est

    if return_info:
        return clip, info
    return clip


# ==================================================
#  ЗАДАЧА 2: Dataset для видеоклипов (PyTorch)
# ==================================================

class VideoDataset(Dataset):
    """
    Dataset, который по каждому пути к видео возвращает один клип фиксированной длины.

    По умолчанию берём клип с начала ролика (start_sec=0).

    Можно добавить свою стратегию сэмплинга, передав функцию sampler, которая
    по duration_sec возвращает start_sec.
    """

    def __init__(
        self,
        video_paths: List[str],
        clip_len: int = 16,
        stride: int = 2,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sampler: Optional[Callable[[float], float]] = None,
    ) -> None:
        super().__init__()
        self.video_paths = list(video_paths)
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.transform = transform
        self.sampler = sampler

    def __len__(self) -> int:
        return len(self.video_paths)

    def _get_duration_sec(self, path: str) -> float:
        with av.open(path) as container:
            stream = container.streams.video[0]
            if stream.duration is None:
                container.seek(0)
                n_frames = 0
                for _ in container.decode(video=0):
                    n_frames += 1
                    if n_frames > 1000:
                        break
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                return n_frames / fps
            return float(stream.duration * stream.time_base)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.video_paths[idx]

        if self.sampler is not None:
            duration = self._get_duration_sec(path)
            start_sec = self.sampler(duration)
        else:
            start_sec = 0.0

        with record_function("decode"):
            clip_np = read_clip(
                path,
                start_sec=start_sec,
                num_frames=self.clip_len,
                stride=self.stride,
                return_info=False,
            )
        clip = torch.from_numpy(clip_np).permute(0, 3, 1, 2).float() / 255.0  # T,C,H,W

        if self.transform is not None:
            with record_function("preprocess_transform"):
                clip = self.transform(clip)

        return clip


# ================================================
#  Модель: компактная 3D-CNN под UCF101
# ================================================

class SmallC3D(nn.Module):
    """
    Простая 3D CNN под action recognition на UCF101.

    Ожидает логически вход (B, T, C, H, W), но умеет сама
    привести любой тензор вида (B, ..., C, H, W) к этому формату.
    """

    def __init__(
        self,
        num_classes: int = 101,
        in_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def _ensure_5d_B_T_C_H_W(self, x: torch.Tensor) -> torch.Tensor:
        """
        Приводит тензор к форме (B, T, C, H, W), предполагая, что
        последние 3 измерения — (C, H, W), а все промежуточные
        сворачиваем в T.
        """
        if x.dim() < 4:
            raise ValueError(f"Ожидалось минимум 4 измерения, получено {x.dim()}")

        # Если уже (B, T, C, H, W) — ничего не делаем
        if x.dim() == 5:
            return x

        # Общий случай: (B, *, C, H, W) или (B, C, H, W)
        B = x.size(0)
        C = x.size(-3)
        H = x.size(-2)
        W = x.size(-1)

        total = x.numel()
        if total % (B * C * H * W) != 0:
            raise ValueError(
                f"Невозможно привести тензор формы {tuple(x.shape)} "
                f"к виду (B, T, C, H, W): размер не делится на B*C*H*W."
            )
        T = total // (B * C * H * W)
        x = x.view(B, T, C, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Приводим к (B, T, C, H, W) независимо от исходной размерности
        x = self._ensure_5d_B_T_C_H_W(x)
        # (B, T, C, H, W) → (B, C, T, H, W) для Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ================================================
#  Вспомогательные: счётчики и throughput
# ================================================

@dataclass
class ThroughputStats:
    total_frames: int
    total_time: float
    fps: float


def measure_throughput(
    dataloader: DataLoader,
    num_epochs: int = 1,
    device: Optional[torch.device] = None,
) -> ThroughputStats:
    """
    Измеряет throughput (кадров/сек) для заданного DataLoader.
    Здесь мы меряем только decode+transfer, без реальной модели.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_frames = 0
    t0 = time.time()
    for _ in range(num_epochs):
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                B, T = batch.shape[:2]
            else:
                raise TypeError("Ожидался тензор батча")

            total_frames += B * T
            batch = batch.to(device, non_blocking=True)

    t1 = time.time()
    total_time = t1 - t0
    fps = total_frames / total_time if total_time > 0 else 0.0
    return ThroughputStats(total_frames=total_frames, total_time=total_time, fps=fps)


# =======================================================
#  ЗАДАЧА 3: параллельная загрузка (num_workers)
# =======================================================

def benchmark_num_workers(
    video_paths: List[str],
    clip_len: int = 16,
    stride: int = 2,
    batch_size: int = 4,
    worker_list: List[int] = (0, 1, 2, 4),
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> List[Tuple[int, ThroughputStats]]:
    """
    Прогоняет несколько значений num_workers и измеряет throughput.
    Возвращает список (num_workers, ThroughputStats).
    """
    results: List[Tuple[int, ThroughputStats]] = []

    base_dataset = VideoDataset(
        video_paths,
        clip_len=clip_len,
        stride=stride,
        transform=transform,
    )

    for nw in worker_list:
        loader = DataLoader(
            base_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            prefetch_factor=2 if nw > 0 else None,
        )
        stats = measure_throughput(loader, num_epochs=1)
        results.append((nw, stats))

    return results


def plot_num_workers_results(results: List[Tuple[int, ThroughputStats]]) -> pd.DataFrame:
    """
    Строит таблицу и график зависимость FPS от num_workers.
    Возвращает DataFrame для удобства анализа.
    """
    data = {
        "num_workers": [nw for nw, _ in results],
        "fps": [stats.fps for _, stats in results],
        "total_frames": [stats.total_frames for _, stats in results],
        "total_time": [stats.total_time for _, stats in results],
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(6, 4))
    plt.plot(df["num_workers"], df["fps"], marker="o")
    plt.xlabel("num_workers")
    plt.ylabel("FPS (кадров/сек)")
    plt.title("Зависимость FPS от num_workers")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df


# ====================================================
#  ЗАДАЧА 4: профилирование этапов пайплайна
# ====================================================

def profile_pipeline(
    dataloader: DataLoader,
    logdir: str,
    device: Optional[torch.device] = None,
    steps: int = 20,
) -> None:
    """
    Профилирует decode (внутри Dataset -> read_clip), preprocess и inference.

    Для корректного профилирования decode имеет смысл использовать num_workers=0,
    чтобы __getitem__ выполнялся в основном процессе.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallC3D().to(device)
    model.eval()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        step = 0
        for batch in dataloader:
            if step >= steps:
                break

            with record_function("preprocess_to_device"):
                batch = batch.to(device, non_blocking=True)

            with record_function("infer"):
                with torch.no_grad():
                    _ = model(batch)

            step += 1
            prof.step()

    print(f"Профиль сохранён в {logdir}. Откройте его через TensorBoard.")


# ======================================================
#  ЗАДАЧА 5 и 9: FPS, jitter, pinned memory, CV(FPS)
# ======================================================

@dataclass
class FPSStats:
    fps_values: List[float]
    mean_fps: float
    std_fps: float
    cv: float


def measure_fps_and_jitter(
    dataloader: DataLoader,
    num_iters: int = 100,
    device: Optional[torch.device] = None,
) -> FPSStats:
    """
    Измеряет мгновенный FPS на каждом батче и считает среднее, std и CV.

    Можно вызывать для разных DataLoader'ов (с/без pin_memory, prefetch)
    и сравнивать стабильность.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fps_values: List[float] = []
    it = iter(dataloader)
    for _ in range(num_iters):
        try:
            t0 = time.time()
            batch = next(it)
        except StopIteration:
            break

        if isinstance(batch, torch.Tensor):
            B, T = batch.shape[:2]
        else:
            raise TypeError("Ожидался тензор батча")

        batch = batch.to(device, non_blocking=True)
        _ = batch.mean()

        t1 = time.time()
        elapsed = t1 - t0
        fps = (B * T) / elapsed if elapsed > 0 else 0.0
        fps_values.append(fps)

    if not fps_values:
        raise RuntimeError("Не удалось измерить ни одного батча (пустой DataLoader?)")

    mean_fps = float(np.mean(fps_values))
    std_fps = float(np.std(fps_values))
    cv = std_fps / mean_fps if mean_fps > 0 else float("inf")

    return FPSStats(fps_values=fps_values, mean_fps=mean_fps, std_fps=std_fps, cv=cv)


def compare_fps_dataloaders(
    loader_base: DataLoader,
    loader_opt: DataLoader,
    num_iters: int = 50,
) -> pd.DataFrame:
    """
    Сравнивает два варианта DataLoader (base vs optimized)
    по mean_fps и cv, строит таблицу и графики.
    """
    stats_base = measure_fps_and_jitter(loader_base, num_iters=num_iters)
    stats_opt = measure_fps_and_jitter(loader_opt, num_iters=num_iters)

    df = pd.DataFrame(
        [
            {
                "variant": "base",
                "mean_fps": stats_base.mean_fps,
                "std_fps": stats_base.std_fps,
                "cv": stats_base.cv,
            },
            {
                "variant": "opt",
                "mean_fps": stats_opt.mean_fps,
                "std_fps": stats_opt.std_fps,
                "cv": stats_opt.cv,
            },
        ]
    )

    plt.figure(figsize=(6, 4))
    plt.bar(df["variant"], df["mean_fps"])
    plt.ylabel("Mean FPS")
    plt.title("Средний FPS (base vs opt)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(df["variant"], df["cv"])
    plt.ylabel("Коэффициент вариации (CV)")
    plt.title("Стабильность FPS (меньше — лучше)")
    plt.tight_layout()
    plt.show()

    return df


# ==================================================================
#  ЗАДАЧА 6: pipeline overlap (decode и infer в разных потоках)
# ==================================================================

class OverlapPipeline:
    """
    Пример перекрытия decode (CPU) и infer (GPU) с помощью threading + Queue.
    """

    def __init__(
        self,
        video_paths: List[str],
        clip_len: int = 16,
        stride: int = 2,
        batch_size: int = 4,
        device: Optional[torch.device] = None,
        max_queue_size: int = 8,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.video_paths = list(video_paths)
        self.clip_len = clip_len
        self.stride = stride
        self.batch_size = batch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.queue: "queue.Queue[Optional[torch.Tensor]]" = queue.Queue(
            maxsize=max_queue_size
        )
        self.model = SmallC3D().to(self.device)
        self.model.eval()
        self.decode_thread: Optional[threading.Thread] = None
        self.stop_flag = False
        self.transform = transform

    def _decode_worker(self) -> None:
        """
        Decode-поток: читает клипы из видео и кладёт в очередь батчи (B,T,C,H,W).
        """
        try:
            clips: List[torch.Tensor] = []
            for path in self.video_paths:
                clip_np = read_clip(
                    path, start_sec=0.0, num_frames=self.clip_len, stride=self.stride
                )
                clip = (
                    torch.from_numpy(clip_np)
                    .permute(0, 3, 1, 2)
                    .float()
                    / 255.0
                )  # T,C,H,W

                if self.transform is not None:
                    clip = self.transform(clip)

                clips.append(clip)

                if len(clips) >= self.batch_size:
                    batch = torch.stack(clips, dim=0)
                    self.queue.put(batch)
                    clips = []

            if clips:
                batch = torch.stack(clips, dim=0)
                self.queue.put(batch)

        finally:
            self.queue.put(None)

    def run(self) -> Tuple[float, float]:
        """
        Запускает перекрытый пайплайн.
        Возвращает (mean_latency, mean_batch_time) по consumer-стороне.
        """
        self.decode_thread = threading.Thread(target=self._decode_worker, daemon=True)
        self.decode_thread.start()

        stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

        latencies: List[float] = []
        batch_times: List[float] = []

        while True:
            try:
                batch = self.queue.get(timeout=10.0)
            except queue.Empty:
                break

            if batch is None:
                break

            t_batch_start = time.time()

            if stream is not None:
                with torch.cuda.stream(stream):
                    t0 = time.time()
                    batch = batch.to(self.device, non_blocking=True)
                    with torch.no_grad():
                        _ = self.model(batch)
                    torch.cuda.synchronize(self.device)
                    t1 = time.time()
            else:
                t0 = time.time()
                batch = batch.to(self.device)
                with torch.no_grad():
                    _ = self.model(batch)
                t1 = time.time()

            latencies.append(t1 - t0)
            batch_times.append(time.time() - t_batch_start)

        if self.decode_thread is not None:
            self.decode_thread.join()

        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        mean_bt = float(np.mean(batch_times)) if batch_times else 0.0
        return mean_lat, mean_bt


# =========================================================
#  ЗАДАЧА 7: аппаратное декодирование (decord vs PyAV)
# =========================================================

@dataclass
class DecodeBenchmarkResult:
    method: str
    total_frames: int
    total_time: float
    fps: float


def benchmark_pyav_decode(path: str, num_frames: Optional[int] = None) -> DecodeBenchmarkResult:
    container = av.open(path)
    total_frames = 0
    t0 = time.time()
    for frame in container.decode(video=0):
        _ = frame.to_rgb().to_ndarray()
        total_frames += 1
        if num_frames is not None and total_frames >= num_frames:
            break
    t1 = time.time()
    container.close()

    total_time = t1 - t0
    fps = total_frames / total_time if total_time > 0 else 0.0
    return DecodeBenchmarkResult("PyAV/CPU", total_frames, total_time, fps)


def benchmark_decord_decode(
    path: str,
    num_frames: Optional[int] = None,
    use_gpu: bool = True,
) -> DecodeBenchmarkResult:
    if not HAS_DECORD:
        raise RuntimeError("decord не установлен, установите `pip install decord`")

    ctx = decord_gpu(0) if use_gpu and torch.cuda.is_available() else decord_cpu(0)

    try:
        vr = VideoReader(path, ctx=ctx)
    except DECORDError as e:
        if "CUDA not enabled" in str(e) and use_gpu:
            print("decord собран без CUDA, переключаюсь на CPU-контекст.")
            ctx = decord_cpu(0)
            vr = VideoReader(path, ctx=ctx)
        else:
            raise

    total_frames = 0
    t0 = time.time()
    for i in range(len(vr)):
        _ = vr[i]
        total_frames += 1
        if num_frames is not None and total_frames >= num_frames:
            break
    t1 = time.time()

    total_time = t1 - t0
    fps = total_frames / total_time if total_time > 0 else 0.0
    method = "decord/GPU" if use_gpu and torch.cuda.is_available() and "gpu" in str(ctx) else "decord/CPU"
    return DecodeBenchmarkResult(method, total_frames, total_time, fps)


def compare_decoders(path: str, num_frames: int = 200, use_gpu: bool = True) -> pd.DataFrame:
    """
    Сравнивает PyAV и decord по скорости декодирования на одном видео.
    Возвращает таблицу с FPS и строит bar chart.
    """
    res_pyav = benchmark_pyav_decode(path, num_frames=num_frames)
    rows = [
        {
            "method": res_pyav.method,
            "frames": res_pyav.total_frames,
            "time": res_pyav.total_time,
            "fps": res_pyav.fps,
        }
    ]

    if HAS_DECORD:
        try:
            res_dec = benchmark_decord_decode(path, num_frames=num_frames, use_gpu=use_gpu)
            rows.append(
                {
                    "method": res_dec.method,
                    "frames": res_dec.total_frames,
                    "time": res_dec.total_time,
                    "fps": res_dec.fps,
                }
            )
        except Exception as e:
            print(f"Ошибка при работе с decord: {e}")

    df = pd.DataFrame(rows)

    plt.figure(figsize=(6, 4))
    plt.bar(df["method"], df["fps"])
    plt.ylabel("FPS")
    plt.title("Сравнение скорости декодирования (PyAV vs decord)")
    plt.tight_layout()
    plt.show()

    return df


# ===================================================
#  ЗАДАЧА 8: оптимизация препроцессинга (CPU vs GPU)
# ===================================================

def make_cpu_transform(image_size: int = 112) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    CPU-препроцессинг: Resize + Normalize.
    clip: (T,C,H,W) -> (T,C,image_size,image_size).
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def transform(clip: torch.Tensor) -> torch.Tensor:
        T = clip.shape[0]
        out_frames = []
        for t in range(T):
            img = clip[t]  # (C,H,W)
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            img = (img - mean) / std
            out_frames.append(img)
        return torch.stack(out_frames, dim=0)

    return transform


def make_gpu_transform(image_size: int = 112) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    GPU-препроцессинг через torchvision.transforms.v2 (если доступен).
    Ожидает тензор на GPU (T,C,H,W).
    """
    if v2 is None:
        raise RuntimeError(
            "torchvision.transforms.v2 недоступен. "
            "Установите torchvision>=0.17 или используйте CPU-препроцессинг."
        )

    transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def gpu_transform(clip: torch.Tensor) -> torch.Tensor:
        return transform(clip)

    return gpu_transform


def benchmark_preprocess_cpu_vs_gpu(
    clip: torch.Tensor,
    n_iters: int = 50,
) -> pd.DataFrame:
    """
    Сравнивает время препроцессинга на CPU и GPU.
    clip: (T,C,H,W) float32 в [0,1] на CPU.
    Возвращает DataFrame и строит bar chart.
    """
    cpu_transform = make_cpu_transform()

    t0 = time.time()
    for _ in range(n_iters):
        _ = cpu_transform(clip)
    t1 = time.time()
    t_cpu = (t1 - t0) / n_iters

    rows = [
        {"device": "cpu", "time_per_iter": t_cpu},
    ]

    if torch.cuda.is_available() and v2 is not None:
        gpu_clip = clip.to("cuda")
        gpu_transform = make_gpu_transform()
        _ = gpu_transform(gpu_clip)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            _ = gpu_transform(gpu_clip)
        torch.cuda.synchronize()
        t1 = time.time()
        t_gpu = (t1 - t0) / n_iters
        rows.append({"device": "cuda", "time_per_iter": t_gpu})
    else:
        print("GPU или torchvision.v2 недоступны, сравнение только для CPU.")

    df = pd.DataFrame(rows)

    plt.figure(figsize=(6, 4))
    plt.bar(df["device"], df["time_per_iter"])
    plt.ylabel("Среднее время препроцессинга, c")
    plt.title("CPU vs GPU препроцессинг")
    plt.tight_layout()
    plt.show()

    return df


# ==========================================================
#  ЗАДАЧА 10: near-real-time пайплайн (decode→infer)
# ==========================================================

@dataclass
class RTPipelineStats:
    mean_fps: float
    p95_latency: float
    jitter: float

def run_near_rt_pipeline(
    path: str,
    clip_len: int = 16,
    stride: int = 2,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    max_frames: Optional[int] = None,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> RTPipelineStats:
    """
    Мини RT-пайплайн для одного видео UCF101:
    - отдельный поток-декодер PyAV → очередь кадров (frame_queue);
    - на consumer-стороне: sliding window → клипы → очередь клипов (clips_queue);
    - опциональный препроцессинг;
    - inference SmallC3D в отдельном CUDA Stream;
    - pinned memory для клипов;
    - измерение FPS, p95 latency и jitter.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallC3D().to(device)
    model.eval()

    frame_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=64)
    stop_event = threading.Event()

    def _decoder_worker() -> None:
        container = av.open(path)
        try:
            for frame in container.decode(video=0):
                if stop_event.is_set():
                    break
                if frame.pts is None:
                    continue
                rgb = frame.to_rgb().to_ndarray()

                try:
                    frame_queue.put(rgb, timeout=1.0)
                except queue.Full:
                    break
        finally:
            container.close()
            try:
                frame_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

    decode_thread = threading.Thread(target=_decoder_worker, daemon=True)
    decode_thread.start()

    frames_buffer: List[np.ndarray] = []
    clips_queue: List[torch.Tensor] = []

    latencies: List[float] = []
    total_frames = 0
    t_global_start = time.time()

    cuda_stream = (
        torch.cuda.Stream(device=device) if device.type == "cuda" else None
    )

    while True:
        try:
            item = frame_queue.get(timeout=5.0)
        except queue.Empty:
            break

        if item is None:
            break

        frames_buffer.append(item)
        total_frames += 1

        if max_frames is not None and total_frames >= max_frames:
            stop_event.set()

        if len(frames_buffer) >= clip_len * stride:
            clip_frames = frames_buffer[::stride][:clip_len]
            frames_buffer = frames_buffer[stride:]

            clip_np = np.stack(clip_frames, axis=0)  # (T, H, W, C)
            clip = (
                torch.from_numpy(clip_np)
                .permute(0, 3, 1, 2)  # (T, C, H, W)
                .float()
                / 255.0
            )

            if transform is not None:
                clip = transform(clip)

            if device.type == "cuda":
                clip = clip.pin_memory()

            clips_queue.append(clip)

        while len(clips_queue) >= batch_size:
            batch = torch.stack(clips_queue[:batch_size], dim=0)  # (B, T, C, H, W)
            clips_queue = clips_queue[batch_size:]

            t0 = time.time()

            if device.type == "cuda":
                assert cuda_stream is not None
                batch = batch.to(device, non_blocking=True)
                with torch.cuda.stream(cuda_stream):
                    with torch.no_grad():
                        _ = model(batch)
                torch.cuda.synchronize(device)
            else:
                batch = batch.to(device)
                with torch.no_grad():
                    _ = model(batch)

            t1 = time.time()
            latencies.append(t1 - t0)

        if max_frames is not None and total_frames >= max_frames:
            break

    stop_event.set()
    if decode_thread.is_alive():
        decode_thread.join(timeout=1.0)

    if not latencies:
        raise RuntimeError("Не удалось измерить latency (нет клипов/батчей).")

    total_time = time.time() - t_global_start
    mean_fps = (total_frames / total_time) if total_time > 0 else 0.0

    lat_arr = np.array(latencies)
    jitter = float(np.std(lat_arr))
    p95 = float(np.percentile(lat_arr, 95.0))

    return RTPipelineStats(mean_fps=mean_fps, p95_latency=p95, jitter=jitter)


def summarize_rt_stats(stats: RTPipelineStats) -> pd.DataFrame:
    """
    Строит простую таблицу и bar chart по метрикам RT-пайплайна.
    """
    df = pd.DataFrame(
        [
            {
                "metric": "mean_fps",
                "value": stats.mean_fps,
            },
            {
                "metric": "p95_latency_ms",
                "value": stats.p95_latency * 1000.0,
            },
            {
                "metric": "jitter_ms",
                "value": stats.jitter * 1000.0,
            },
        ]
    )

    plt.figure(figsize=(6, 4))
    plt.bar(df["metric"], df["value"])
    plt.title("Метрики near-RT пайплайна")
    plt.tight_layout()
    plt.show()

    return df


# ==========================================================
#  ЗАДАЧИ 1–10: отдельные функции-демки для ноутбука
# ==========================================================

def run_task1_read_clip_demo(video_paths: Optional[List[str]] = None) -> None:
    """
    Задача 1: показать работу read_clip на одном видео UCF101.
    """
    if video_paths is None:
        video_paths = get_ucf101_videos_kaggle(limit_videos=8)
    path = video_paths[0]
    clip, info = read_clip(path, num_frames=8, stride=2, return_info=True)
    print("Видео:", path)
    print("clip shape:", clip.shape)
    print("indices:", info["frame_indices"])
    print("timestamps:", info["timestamps"])
    print("fps_est:", info["fps_est"])


def run_task2_3_dataloader_num_workers_demo(video_paths: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Задачи 2–3: создать VideoDataset, DataLoader и сравнить num_workers.
    Возвращает DataFrame с результатами.
    """
    if video_paths is None:
        video_paths = get_ucf101_videos_kaggle(limit_videos=32)

    transform = make_cpu_transform()
    results = benchmark_num_workers(
        video_paths,
        clip_len=16,
        stride=2,
        batch_size=4,
        worker_list=[0, 1, 2, 4],
        transform=transform,
    )

    df = plot_num_workers_results(results)
    print(df)
    return df


def run_task4_profile_pipeline(video_paths: Optional[List[str]] = None, logdir: str = "logs/profile_task4") -> None:
    """
    Задача 4: профилирование decode+preprocess+infer на нескольких батчах.
    """
    if video_paths is None:
        video_paths = get_ucf101_videos_kaggle(limit_videos=8)

    transform = make_cpu_transform()
    dataset = VideoDataset(video_paths, clip_len=16, stride=2, transform=transform)
    loader = DataLoader(dataset, batch_size=2, num_workers=0, pin_memory=True)
    profile_pipeline(loader, logdir=logdir)


def run_task4_profile_no_tensorboard(
    video_paths: List[str],
    clip_len: int = 16,
    stride: int = 2,
    batch_size: int = 2,
    steps: int = 20,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Задача 4 (альтернатива): Профилирует decode + preprocess + infer и выводит результаты
    прямо в ноутбуке, без TensorBoard.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Фиксируем сиды для воспроизводимости
    set_global_seed(42)

    transform = make_cpu_transform()
    dataset = VideoDataset(video_paths, clip_len=clip_len, stride=stride, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    model = SmallC3D().to(device)
    model.eval()

    print("Начинаю профилирование...")

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ] if device.type == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:

        it = iter(loader)
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                break

            # decode уже обёрнут в record_function("decode") внутри VideoDataset.__getitem__

            # Явно профилируем препроцессинг
            with record_function("preprocess_to_device"):
                batch = batch.to(device, non_blocking=True)

            # Явно профилируем инференс
            with record_function("infer"):
                with torch.no_grad():
                    _ = model(batch)

            if device.type == "cuda":
                torch.cuda.synchronize()

    print("\nПрофилирование завершено.\n")

    cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    cuda_table = (
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        if device.type == "cuda"
        else "CUDA недоступно"
    )

    # Парсим для DataFrame
    rows = []
    for evt in prof.key_averages():
        cpu_ms = evt.cpu_time_total / 1e6
        raw_cuda_time = getattr(evt, "cuda_time_total", 0.0)
        cuda_ms = (raw_cuda_time / 1e6) if (device.type == "cuda" and raw_cuda_time is not None) else 0.0

        rows.append(
            {
                "operator": evt.key,
                "cpu_total_ms": cpu_ms,
                "cuda_total_ms": cuda_ms,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["cpu_total_ms", "cuda_total_ms"], ascending=False).reset_index(drop=True)

    print("ТОП операторов по CPU-времени:")
    print(cpu_table)

    if device.type == "cuda":
        print("\nТОП операторов по CUDA-времени:")
        print(cuda_table)

    top_cpu = df.iloc[0]["operator"]
    top_cpu_time = df.iloc[0]["cpu_total_ms"]

    if device.type == "cuda" and (df["cuda_total_ms"] > 0).any():
        top_cuda = df.sort_values("cuda_total_ms", ascending=False).iloc[0]
        top_cuda_op = top_cuda["operator"]
        top_cuda_time = top_cuda["cuda_total_ms"]
    else:
        top_cuda_op = "N/A"
        top_cuda_time = 0.0

    print("\n=== КРАТКИЕ ВЫВОДЫ ===")
    print(f"- Самый дорогой CPU-оператор: **{top_cpu}** (~{top_cpu_time:.2f} ms)")
    if device.type == "cuda":
        print(f"- Самый дорогой GPU-оператор: **{top_cuda_op}** (~{top_cuda_time:.2f} ms)")

    print("- Если CPU операторы занимают больше времени, чем CUDA, значит GPU простаивает из-за медленной подачи данных.")
    print("- Если cuda_time_total массово превышает cpu_time_total, модель является основным bottleneck.")

    # --- Отношение L_dec : L_prep : L_inf по CPU-времени ---
    def _cpu_ms(op: str) -> float:
        rows_op = df[df["operator"] == op]
        return float(rows_op["cpu_total_ms"].sum()) if not rows_op.empty else 0.0

    dec_ms = _cpu_ms("decode")
    prep_ms = _cpu_ms("preprocess_to_device")
    inf_ms = _cpu_ms("infer")
    core_total = dec_ms + prep_ms + inf_ms

    if core_total > 0.0:
        r_dec = dec_ms / core_total
        r_prep = prep_ms / core_total
        r_inf = inf_ms / core_total
        print(
            f"\nОтношение L_dec : L_prep : L_inf "
            f"≈ {r_dec:.2f} : {r_prep:.2f} : {r_inf:.2f} "
            "(по CPU-времени decode / preprocess_to_device / infer)"
        )
    else:
        print("\nНе удалось посчитать L_dec : L_prep : L_inf (суммарное время = 0).")

    return df
    

def run_task5_9_fps_jitter_demo(video_paths: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Задачи 5 и 9: сравнение DataLoader с/без pin_memory/prefetch по mean_fps и CV.
    """
    set_global_seed(42)
    
    if video_paths is None:
        video_paths = get_ucf101_videos_kaggle(limit_videos=32)

    transform = make_cpu_transform()
    dataset = VideoDataset(video_paths, clip_len=8, stride=2, transform=transform)

    loader_base = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=False)
    loader_opt = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    df = compare_fps_dataloaders(loader_base, loader_opt, num_iters=20)
    print(df)
    return df


def run_task6_overlap_pipeline_demo(video_paths: Optional[List[str]] = None) -> Tuple[float, float]:
    """
    Задача 6: запустить OverlapPipeline и вывести среднюю latency и время батча.
    """
    set_global_seed(42)
    
    if video_paths is None:
        video_paths = get_ucf101_videos_kaggle(limit_videos=16)

    transform = make_cpu_transform()
    overlap = OverlapPipeline(
        video_paths,
        clip_len=8,
        stride=2,
        batch_size=2,
        transform=transform,
    )
    mean_lat, mean_bt = overlap.run()
    print(f"mean_latency = {mean_lat*1000:.2f} ms, mean_batch_time = {mean_bt*1000:.2f} ms")
    return mean_lat, mean_bt


def run_task7_decoder_compare_demo(video_paths: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Задача 7: сравнить PyAV и decord на одном видео UCF101.
    """
    set_global_seed(42)
    
    if video_paths is None:
        video_paths = get_ucf101_videos_kaggle(limit_videos=8)
    path = video_paths[0]
    df = compare_decoders(path, num_frames=200, use_gpu=True)
    print(df)
    return df


def run_task8_preprocess_cpu_gpu_demo(video_paths: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Задача 8: сравнить препроцессинг CPU vs GPU на одном клипе из UCF101.
    """
    set_global_seed(42)
    
    if video_paths is None:
        video_paths = get_ucf101_videos_kaggle(limit_videos=8)
    path = video_paths[0]
    clip_np = read_clip(path, num_frames=16, stride=2, return_info=False)
    clip = torch.from_numpy(clip_np).permute(0, 3, 1, 2).float() / 255.0
    df = benchmark_preprocess_cpu_vs_gpu(clip, n_iters=30)
    print(df)
    return df


def run_task10_near_rt_demo(video_paths: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Задача 10: запустить near-real-time пайплайн на одном видео UCF101 и
    построить таблицу/график метрик.
    """
    set_global_seed(42)
    
    if video_paths is None:
        video_paths = get_ucf101_videos_kaggle(limit_videos=8)
    path = video_paths[0]

    transform = make_cpu_transform()
    stats = run_near_rt_pipeline(
        path,
        clip_len=8,
        stride=2,
        batch_size=2,
        max_frames=300,
        transform=transform,
    )
    print(stats)
    df = summarize_rt_stats(stats)
    print(df)
    return df


# ==========================================================
#  Мини-ДЗ: сравнение offline vs near-real-time режимов
# ==========================================================

@dataclass
class PipelineModeStats:
    mode: str
    mean_fps: float      # средний FPS (по всему прогону)
    fps_cv: float        # коэффициент вариации FPS (стабильность)
    mean_latency: float  # средняя latency (сек)
    p95_latency: float   # 95-й перцентиль latency (сек)
    jitter: float        # std latency (сек)


def run_offline_pipeline_stats(
    video_paths: List[str],
    clip_len: int = 16,
    stride: int = 2,
    batch_size: int = 4,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    max_batches: int = 20,
) -> PipelineModeStats:
    """
    OFFLINE-режим:
    - VideoDataset + DataLoader (файловый ввод),
    - SmallC3D,
    - измеряем latency и FPS на уровне батчей.

    max_batches ограничивает число батчей, чтобы не гонять всё UCF101.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if transform is None:
        transform = make_cpu_transform()

    dataset = VideoDataset(video_paths, clip_len=clip_len, stride=stride, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    model = SmallC3D().to(device)
    model.eval()

    latencies: List[float] = []
    fps_values: List[float] = []
    total_frames = 0
    total_time = 0.0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            if not isinstance(batch, torch.Tensor):
                raise TypeError("Ожидался тензор батча")

            B, T = batch.shape[:2]
            frames = B * T

            t0 = time.time()
            batch = batch.to(device, non_blocking=True)
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.time()

            elapsed = t1 - t0
            total_frames += frames
            total_time += elapsed
            latencies.append(elapsed)
            fps_values.append(frames / elapsed if elapsed > 0 else 0.0)

    lat_arr = np.array(latencies)
    fps_arr = np.array(fps_values)

    mean_fps = float(total_frames / total_time) if total_time > 0 else 0.0
    fps_cv = float(fps_arr.std() / fps_arr.mean()) if fps_arr.size > 0 and fps_arr.mean() > 0 else float("inf")
    mean_lat = float(lat_arr.mean()) if lat_arr.size > 0 else 0.0
    p95 = float(np.percentile(lat_arr, 95.0)) if lat_arr.size > 0 else 0.0
    jitter = float(lat_arr.std()) if lat_arr.size > 0 else 0.0

    return PipelineModeStats(
        mode="offline",
        mean_fps=mean_fps,
        fps_cv=fps_cv,
        mean_latency=mean_lat,
        p95_latency=p95,
        jitter=jitter,
    )


def run_near_rt_pipeline_stats(
    path: str,
    clip_len: int = 16,
    stride: int = 2,
    batch_size: int = 4,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    max_frames: Optional[int] = 1000,
) -> PipelineModeStats:
    """
    NEAR-REAL-TIME режим:
    - потоковый decode одного видео (как из камеры),
    - sliding window → клипы,
    - SmallC3D,
    - измеряем latency и FPS на уровне батчей.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if transform is None:
        transform = make_cpu_transform()

    container = av.open(path)
    _ = container.streams.video[0]  # stream нужен для decode, но таймбейс тут не используем

    model = SmallC3D().to(device)
    model.eval()

    frames_buffer: List[np.ndarray] = []
    clips_queue: List[torch.Tensor] = []

    latencies: List[float] = []
    fps_values: List[float] = []
    total_frames = 0
    total_time = 0.0

    with torch.no_grad():
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            rgb = frame.to_rgb().to_ndarray()
            frames_buffer.append(rgb)
            total_frames += 1

            # sliding window по кадрам → клипы
            if len(frames_buffer) >= clip_len * stride:
                clip_frames = frames_buffer[::stride][:clip_len]
                frames_buffer = frames_buffer[stride:]

                clip_np = np.stack(clip_frames, axis=0)
                clip = (
                    torch.from_numpy(clip_np)
                    .permute(0, 3, 1, 2)
                    .float()
                    / 255.0
                )  # T,C,H,W

                clip = transform(clip)
                clips_queue.append(clip)

            # как только набрали батч клипов – считаем latency/FPS
            if len(clips_queue) >= batch_size:
                batch = torch.stack(clips_queue[:batch_size], dim=0)
                clips_queue = clips_queue[batch_size:]

                B, T = batch.shape[:2]
                frames = B * T

                t0 = time.time()
                batch = batch.to(device, non_blocking=True)
                _ = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1 = time.time()

                elapsed = t1 - t0
                total_time += elapsed
                latencies.append(elapsed)
                fps_values.append(frames / elapsed if elapsed > 0 else 0.0)

            if max_frames is not None and total_frames >= max_frames:
                break

    container.close()

    lat_arr = np.array(latencies)
    fps_arr = np.array(fps_values)

    mean_fps = float(total_frames / total_time) if total_time > 0 else 0.0
    fps_cv = float(fps_arr.std() / fps_arr.mean()) if fps_arr.size > 0 and fps_arr.mean() > 0 else float("inf")
    mean_lat = float(lat_arr.mean()) if lat_arr.size > 0 else 0.0
    p95 = float(np.percentile(lat_arr, 95.0)) if lat_arr.size > 0 else 0.0
    jitter = float(lat_arr.std()) if lat_arr.size > 0 else 0.0

    return PipelineModeStats(
        mode="near_real_time",
        mean_fps=mean_fps,
        fps_cv=fps_cv,
        mean_latency=mean_lat,
        p95_latency=p95,
        jitter=jitter,
    )


def run_minidz_offline_vs_rt(
    video_paths: Optional[List[str]] = None,
    clip_len: int = 16,
    stride: int = 2,
    batch_size: int = 4,
    max_batches_offline: int = 20,
    max_frames_rt: int = 1000,
) -> pd.DataFrame:
    """
    Мини-ДЗ:
    - запускает offline и near-real-time режимы,
    - сравнивает latency и стабильность FPS,
    - возвращает таблицу и рисует графики.

    В качестве "потока" для RT используем первое видео из списка.
    """
    set_global_seed(42)
    
    if video_paths is None:
        video_paths = get_ucf101_videos_kaggle(limit_videos=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = make_cpu_transform()

    # OFFLINE: несколько файлов через Dataset/DataLoader
    offline_stats = run_offline_pipeline_stats(
        video_paths=video_paths[:16],
        clip_len=clip_len,
        stride=stride,
        batch_size=batch_size,
        transform=transform,
        device=device,
        max_batches=max_batches_offline,
    )

    # NEAR-RT: поток с одного видео
    rt_stats = run_near_rt_pipeline_stats(
        path=video_paths[0],
        clip_len=clip_len,
        stride=stride,
        batch_size=batch_size,
        transform=transform,
        device=device,
        max_frames=max_frames_rt,
    )

    rows = [
        {
            "mode": offline_stats.mode,
            "mean_fps": offline_stats.mean_fps,
            "fps_cv": offline_stats.fps_cv,
            "mean_latency_ms": offline_stats.mean_latency * 1000.0,
            "p95_latency_ms": offline_stats.p95_latency * 1000.0,
            "jitter_ms": offline_stats.jitter * 1000.0,
        },
        {
            "mode": rt_stats.mode,
            "mean_fps": rt_stats.mean_fps,
            "fps_cv": rt_stats.fps_cv,
            "mean_latency_ms": rt_stats.mean_latency * 1000.0,
            "p95_latency_ms": rt_stats.p95_latency * 1000.0,
            "jitter_ms": rt_stats.jitter * 1000.0,
        },
    ]
    df = pd.DataFrame(rows)

    # Графики: mean FPS и CV
    plt.figure(figsize=(6, 4))
    plt.bar(df["mode"], df["mean_fps"])
    plt.ylabel("Mean FPS")
    plt.title("Сравнение mean FPS: offline vs near-real-time")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(df["mode"], df["fps_cv"])
    plt.ylabel("Коэффициент вариации FPS (CV)")
    plt.title("Стабильность FPS (меньше — лучше)")
    plt.tight_layout()
    plt.show()

    # Краткие комментарии (для отчёта)
    off = offline_stats
    rt = rt_stats

    print("Краткие выводы:")
    if off.mean_fps > rt.mean_fps:
        print(f"- Offline-режим быстрее по среднему FPS ({off.mean_fps:.1f} > {rt.mean_fps:.1f}).")
    else:
        print(f"- Near-real-time режим быстрее по среднему FPS ({rt.mean_fps:.1f} ≥ {off.mean_fps:.1f}).")

    if off.fps_cv < rt.fps_cv:
        print(f"- FPS стабильнее в offline-режиме (CV={off.fps_cv:.3f} < {rt.fps_cv:.3f}).")
    else:
        print(f"- FPS стабильнее в near-real-time (CV={rt.fps_cv:.3f} ≤ {off.fps_cv:.3f}).")

    if off.mean_latency < rt.mean_latency:
        print(
            f"- Средняя latency ниже в offline-режиме "
            f"({off.mean_latency*1000:.1f} ms < {rt.mean_latency*1000:.1f} ms)."
        )
    else:
        print(
            f"- Средняя latency ниже в near-real-time "
            f"({rt.mean_latency*1000:.1f} ms ≤ {off.mean_latency*1000:.1f} ms)."
        )

    return df
