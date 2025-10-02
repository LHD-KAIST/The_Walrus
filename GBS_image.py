"""Gaussian Boson Sampling 기반 간소화 이미지 분류 실험.

논문 "Enhanced Image Recognition Using Gaussian Boson Sampling"의 실험 흐름을
참고해 메모리 제약(약 1.5GB) 안에서 재현 가능한 소규모 파이프라인을 구현한다.

주요 단계
---------
1. 소형 이미지 데이터셋(Scikit-learn digits)을 불러오고 PCA로 차원을 축소한다.
2. PCA 특징을 정규화/시그모이드 변환 후 GBS 회로의 squeezing 파라미터로 매핑한다.
3. Strawberry Fields Fock 백엔드를 사용해 GBS 샘플(포톤 카운트)을 생성한다.
4. 원본 PCA 특징과 GBS로부터 얻은 특성을 결합해 선형 분류기
   (Perceptron), Gaussian ELM, GRVFL 변형을 학습한다.

실제 Jiuzhang 실험에 비해 모드 수와 샘플 수, cutoff 등을 크게 줄였으며,
초기 squeeze 크기를 낮게 설정해 Fock 차원을 4로 제한함으로써 메모리 사용을
억제한다. 각 구성 요소는 클래스화하여 다른 데이터셋이나 하이퍼파라미터에도
쉽게 적용할 수 있도록 했다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.utils import random_interferometer


# ------------------------------ 유틸리티 함수 ------------------------------ #

def sigmoid(x: ArrayLike) -> np.ndarray:
    """수치 안정성을 고려한 시그모이드."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def one_hot(labels: np.ndarray) -> np.ndarray:
    """정수형 라벨을 원-핫으로 변환."""
    classes = np.unique(labels)
    mapping = {label: idx for idx, label in enumerate(classes)}
    y = np.array([mapping[label] for label in labels], dtype=int)
    eye = np.eye(len(classes))
    return eye[y]


# ----------------------------- GBS 구성 파트 ------------------------------ #

@dataclass
class GBSConfig:
    modes: int = 5 # PCA 출력 차원이자 mode 수 ** Checkpoint : Overtraining modes 수?
    cutoff: int = 4 # 각 모드의 최대 photon 수 
    shots_per_sample: int = 7 # 엔진 반복 실행 횟수수
    max_squeezing: float = 0.35 # squeezing하는 정도
    interferometer_seed: int = 42
    backend: str = "fock"


class GBSFeatureExtractor:
    """PCA 특징을 Gaussian Boson Sampling 결과로 변환."""

    def __init__(self, config: GBSConfig) -> None:
        self.config = config
        if config.interferometer_seed is not None:
            np.random.seed(config.interferometer_seed)
        self._unitary = random_interferometer(config.modes)

    def _build_program(self, squeezings: np.ndarray) -> sf.Program:
        prog = sf.Program(self.config.modes)
        with prog.context as q:
            for idx, r in enumerate(squeezings):
                ops.Sgate(r) | q[idx]
            ops.Interferometer(self._unitary) | q
            ops.MeasureFock() | q
        return prog

    def sample_counts(self, squeezings: np.ndarray) -> np.ndarray:
        shots = max(1, int(self.config.shots_per_sample))
        samples = []
        for _ in range(shots):
            prog = self._build_program(squeezings)
            eng = sf.Engine(
                self.config.backend,
                backend_options={"cutoff_dim": self.config.cutoff},
            )
            result = eng.run(prog)
            samples.append(result.samples[0])
        samples_arr = np.asarray(samples, dtype=np.int64)
        mean_counts = samples_arr.mean(axis=0)
        return mean_counts

    def batch_sample(self, squeezing_matrix: np.ndarray) -> np.ndarray:
        features = [self.sample_counts(row) for row in squeezing_matrix]
        counts = np.vstack(features)
        counts_sum = counts.sum(axis=1, keepdims=True)
        counts_sum[counts_sum == 0] = 1.0
        normalized = counts / counts_sum
        return normalized


# ---------------------------- 분류기 구현 파트 ---------------------------- #

@dataclass
class GELMConfig:
    hidden_units: int = 64
    reg: float = 1e-3
    random_state: int = 0


class GaussianELM:
    """Gaussian 활성화를 사용하는 간단한 Extreme Learning Machine."""

    def __init__(self, input_dim: int, config: GELMConfig) -> None:
        self.config = config
        rng = np.random.RandomState(config.random_state)
        self.W = rng.normal(scale=0.5, size=(input_dim, config.hidden_units))
        self.b = rng.uniform(-1.0, 1.0, size=(config.hidden_units,))
        self.beta: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        squared_norm = np.sum((X[:, :, None] - self.W[None, :, :]) ** 2, axis=1)
        return np.exp(-squared_norm + self.b)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianELM":
        H = self._hidden(X)
        T = one_hot(y)
        reg_eye = self.config.reg * np.eye(H.shape[1])
        self.beta = np.linalg.solve(H.T @ H + reg_eye, H.T @ T)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None or self.classes_ is None:
            raise RuntimeError("모델이 아직 학습되지 않았습니다.")
        H = self._hidden(X)
        logits = H @ self.beta
        indices = np.argmax(logits, axis=1)
        return self.classes_[indices]


@dataclass
class GRVFLConfig:
    hidden_units: int = 48
    reg: float = 1e-3
    random_state: int = 1


class GRVFLClassifier:
    """입력 직접 연결이 포함된 Random Vector Functional Link 분류기."""

    def __init__(self, input_dim: int, config: GRVFLConfig) -> None:
        self.config = config
        rng = np.random.RandomState(config.random_state)
        self.W = rng.normal(scale=0.4, size=(input_dim, config.hidden_units))
        self.b = rng.uniform(-0.5, 0.5, size=(config.hidden_units,))
        self.beta: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W + self.b)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GRVFLClassifier":
        H = self._hidden(X)
        augmented = np.hstack([X, H])
        T = one_hot(y)
        reg_eye = self.config.reg * np.eye(augmented.shape[1])
        self.beta = np.linalg.solve(augmented.T @ augmented + reg_eye, augmented.T @ T)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None or self.classes_ is None:
            raise RuntimeError("모델이 아직 학습되지 않았습니다.")
        H = self._hidden(X)
        augmented = np.hstack([X, H])
        logits = augmented @ self.beta
        indices = np.argmax(logits, axis=1)
        return self.classes_[indices]


# ----------------------------- 전체 실험 파이프라인 ----------------------------- #

@dataclass
class ExperimentConfig:
    max_train_samples: int = 80
    max_test_samples: int = 40
    test_size: float = 0.3
    random_state: int = 7
    perceptron_max_iter: int = 2000
    gbs: GBSConfig = field(default_factory=GBSConfig)
    gelm: GELMConfig = field(default_factory=GELMConfig)
    grvfl: GRVFLConfig = field(default_factory=GRVFLConfig)


class GBSPipeline:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.gbs.modes, random_state=config.random_state)
        self.gbs_extractor = GBSFeatureExtractor(config.gbs)

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        digits = load_digits()
        X = digits.data.astype(np.float64) / 16.0
        y = digits.target
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        if self.config.max_train_samples:
            X_train = X_train[: self.config.max_train_samples]
            y_train = y_train[: self.config.max_train_samples]
        if self.config.max_test_samples:
            X_test = X_test[: self.config.max_test_samples]
            y_test = y_test[: self.config.max_test_samples]
        return X_train, X_test, y_train, y_test

    def _pca_embedding(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        X_train_sig = sigmoid(X_train_pca)
        X_test_sig = sigmoid(X_test_pca)
        return X_train_sig, X_test_sig

    def _squeezing_map(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X * self.config.gbs.max_squeezing, 0.0, self.config.gbs.max_squeezing)

    def evaluate(self, verbose: bool = True) -> dict:
        X_train, X_test, y_train, y_test = self._prepare_data()
        X_train_sig, X_test_sig = self._pca_embedding(X_train, X_test)
        squeezings_train = self._squeezing_map(X_train_sig)
        squeezings_test = self._squeezing_map(X_test_sig)
        if verbose:
            print("GBS 샘플 생성 중 (훈련 세트)...")
        gbs_train = self.gbs_extractor.batch_sample(squeezings_train)
        if verbose:
            print("GBS 샘플 생성 중 (테스트 세트)...")
        gbs_test = self.gbs_extractor.batch_sample(squeezings_test)
        train_features = np.hstack([X_train_sig, gbs_train])
        test_features = np.hstack([X_test_sig, gbs_test])
        if verbose:
            print("Perceptron 학습/평가...")
        perceptron = Perceptron(max_iter=self.config.perceptron_max_iter, random_state=self.config.random_state)
        perceptron.fit(train_features, y_train)
        perceptron_pred = perceptron.predict(test_features)
        perceptron_acc = accuracy_score(y_test, perceptron_pred)
        if verbose:
            print(f"Perceptron 정확도: {perceptron_acc * 100:.2f}%")
        if verbose:
            print("GELM 학습/평가...")
        gelm = GaussianELM(input_dim=train_features.shape[1], config=self.config.gelm)
        gelm.fit(train_features, y_train)
        gelm_pred = gelm.predict(test_features)
        gelm_acc = accuracy_score(y_test, gelm_pred)
        if verbose:
            print(f"GELM 정확도: {gelm_acc * 100:.2f}%")
        if verbose:
            print("GRVFL 학습/평가...")
        grvfl = GRVFLClassifier(input_dim=train_features.shape[1], config=self.config.grvfl)
        grvfl.fit(train_features, y_train)
        grvfl_pred = grvfl.predict(test_features)
        grvfl_acc = accuracy_score(y_test, grvfl_pred)
        if verbose:
            print(f"GRVFL 정확도: {grvfl_acc * 100:.2f}%")
        metrics = {"perceptron": perceptron_acc, "gelm": gelm_acc, "grvfl": grvfl_acc}
        return metrics

    def run(self) -> None:
        self.evaluate(verbose=True)


def main() -> None:
    config = ExperimentConfig()
    pipeline = GBSPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
