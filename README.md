# 📈 Time Series Forecasting Benchmark
> **Machine Learning Project 5**: *"Prognozowanie szeregów czasowych: czy mechanizm uwagi jest potrzebny?"*  
> *(Time Series Forecasting: Is the Attention Mechanism Needed?)*

Witaj w repozytorium zespołu projektowego! Ten dokument zawiera wszystko, co niezbędne, aby w kilka minut uruchomić środowisko, zrozumieć architekturę projektu i efektywnie współtworzyć kod.

---

## 🎯 Cel Projektu i Pytanie Badawcze

Projekt ma na celu empiryczną weryfikację głośnej tezy z artykułu *DLinear* (Zeng et al., 2022): **czy złożone mechanizmy uwagi w architekturach Transformerowych są rzeczywiście niezbędne w szeregach czasowych**, czy też proste modele liniowe z dekompozycją trendu i sezonowości (DLinear, NLinear) osiągają zbliżoną lub lepszą dokładność przy wielokrotnie niższym koszcie obliczeniowym?

W projekcie porównujemy 4 paradygmaty modelowania:
1. **Klasyczne bazy statystyczne**: `AutoARIMA` / `SARIMA` oraz `SeasonalNaive` (poprzez `statsforecast`).
2. **Głębokie modele bez uwagi (MLP / Linear Family)**: `N-BEATS`, `N-HiTS` oraz `DLinear` (poprzez `neuralforecast`).
3. **Nowoczesne Transformery (Attention Family)**: `PatchTST` (sub-series patching) oraz `iTransformer` (odwrócona uwaga między zmiennymi).
4. **Modele Fundacyjne (Foundation Models / Zero-Shot)**: `Chronos-Bolt` (Amazon, ewaluacja bez dotrenowywania na zbiorze).

Do strojenia modeli używamy **Optuna**, a przewagi modeli weryfikujemy testami istotności statystycznej (**Diebold-Mariano**, **Wilcoxon**, **Friedman + Nemenyi**).

Szczegółowy harmonogram prac i podział na fazy znajduje się w [**`docs/PROJECT_PLAN.md`**](file:///Users/rzadkows/coding/studies/machine-learning/project/docs/PROJECT_PLAN.md).

---

## ⚡ Szybki Start (Onboarding w 3 Krokach)

W projekcie korzystamy z nowoczesnego menedżera pakietów [`uv`](https://docs.astral.sh/uv/) (nie używamy klasycznego `pip` ani `conda`).

### Krok 1: Instalacja `uv` (jeśli jeszcze nie posiadasz)
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Krok 2: Instalacja środowiska i zależności
Będąc w folderze `project/`, uruchom:
```bash
cd project
uv sync
```
`uv sync` automatycznie:
- Tworzy dedykowane wirtualne środowisko `.venv` oparte na Pythonie 3.12.
- Pobiera i instaluje PyTorch, NeuralForecast, StatsForecast, scikit-learn, ruff itd.
- Instaluje pakiet `ts_forecasting_benchmark` w **trybie edytowalnym** (`pip install -e .`), co pozwala importować własne moduły z dowolnego miejsca w projekcie.

### Krok 3: Weryfikacja danych i bibliotek
Uruchom skrypt diagnostyczny, aby potwierdzić poprawność instalacji i obecność zbiorów:
```bash
uv run python scripts/verify_datasets.py
```
Powinieneś zobaczyć zielone komunikaty `✅ OK` dla wszystkich 7 zbiorów danych oraz dla loadera Monash.

### Pilot benchmark

Pilot porównuje `SeasonalNaive`, `AutoARIMA`, `N-HiTS`, `DLinear`, `N-BEATS`,
`PatchTST`, `iTransformer` i `Chronos-Bolt` na ETTh1 (target `OT`, H=96, L=96).
Modele neuronowe są konfigurowane w `src/ts_forecasting_benchmark/models/neural.py`.

```bash
uv run python scripts/run_pilot_benchmark.py --accelerator cpu --max-steps 50
```

Na Apple Silicon użyj `--accelerator mps`, a na karcie NVIDIA `--accelerator cuda`.
Bez tej opcji urządzenie jest wybierane automatycznie. `--max-steps` ustawia liczbę
kroków treningu każdego modelu neuronowego (dotychczasowe `--nhits-steps` nadal działa).
Prognozy dotyczą pierwszych 96 punktów testu i korzystają z historii do końca walidacji.
Metryki i wykres trafiają do `results/metrics/` i `results/figures/`.

---

## 📂 Gdzie co znaleźć? (Przewodnik po Strukturze)

```text
project/
├── configs/                  # [KONFIGURACJE] Pliki YAML/JSON z hiperparametrami eksperymentów
├── data/
│   ├── raw/                  # [DANE] Pobrane surowe zbiory benchmarkowe (.csv)
│   └── processed/            # [DANE] Zbiory po standaryzacji / podziale (ignorowane w git)
├── docs/
│   ├── PROJECT_PLAN.md       # [DOKUMENTACJA] Szczegółowy harmonogram i pytania do prowadzącego
│   ├── PAPERS_SUMMARY.md     # [DOKUMENTACJA] Ekstrakty i kluczowe wnioski z 6 artykułów
│   └── papers/               # [LITERATURA] Pobrane artykuły źródłowe w PDF
├── notebooks/                # [NOTEBOOKI] Notatniki Jupyter do EDA, analizy predykcji i wykresów
├── results/
│   ├── figures/              # [WYNIKI] Wykresy porównawcze i mapy uwagi
│   ├── logs/                 # [WYNIKI] Logi z treningów
│   └── metrics/              # [WYNIKI] Zapisane metryki (MSE, MAE, czas treningu w CSV)
├── scripts/
│   ├── download_datasets.py  # [SKRYPTY] CLI do pobierania zbiorów (np. na świeżej maszynie)
│   └── verify_datasets.py    # [SKRYPTY] Skrypt testujący integralność danych i loaderów
└── src/ts_forecasting_benchmark/ # [KOD ŹRÓDŁOWY] Główny pakiet Python
    ├── data/                 # Loadery danych (PyTorch Dataset, split train/val/test, Monash .tsf)
    ├── models/               # Architektury modeli (DLinear, NLinear, PatchTST, iTransformer)
    ├── evaluation/           # Obliczanie metryk (MSE, MAE, testy istotności statystycznej)
    └── utils/                # Reprodukowalność (seedowanie) i wybór akceleratora (MPS/CUDA/CPU)
```

---

## 🛠️ Jak pracować z projektem? (Workflow dla Zespołu)

### 1. Uruchamianie czegokolwiek z poprawnym środowiskiem
Zamiast aktywować środowisko za każdym razem przez `source .venv/bin/activate`, używaj prefiksu `uv run`:
```bash
uv run python sciezka/do/skryptu.py
uv run jupyter lab
```

### 2. Standardy jakości kodu i Linter (`ruff`)
Przed każdym zatwierdzeniem zmian w Git (`git commit`) uruchom linter:
```bash
# Sprawdzenie błędów i stylu
uv run ruff check .

# Automatyczna naprawa i sortowanie importów
uv run ruff check --fix .

# Automatyczne sformatowanie kodu (zgodnie z PEP 8 / black-style)
uv run ruff format .
```
> [!TIP]
> Jeśli używasz VS Code lub Cursor/Antigravity IDE, plik [`.vscode/settings.json`](file:///Users/rzadkows/coding/studies/machine-learning/project/.vscode/settings.json) automatycznie formatuje kod przy każdym zapisie (`Cmd+S` / `Ctrl+S`).

### 3. Jak wczytać dane w swoim skrypcie lub notebooku?
Wszystkie narzędzia danych są gotowe do importu:

```python
from ts_forecasting_benchmark.data import load_benchmark_dataframe, TimeSeriesBenchmarkDataset
from ts_forecasting_benchmark.utils import set_seed, get_device
import torch
from torch.utils.data import DataLoader

# 1. Zapewnienie powtarzalności i wybór urządzenia (MPS na Macu / CUDA na serwerze / CPU)
set_seed(42)
device = get_device()

# 2. Wczytanie ramki danych
df = load_benchmark_dataframe("data/raw/ETTh1.csv")

# 3. Utworzenie PyTorch Dataset (okno lookback L=96, horyzont prognozy H=96)
# Scaler jest automatycznie dopasowywany TYLKO na zbiorze treningowym (zero data leakage!)
train_ds = TimeSeriesBenchmarkDataset(df, seq_len=96, pred_len=96, split="train")
test_ds  = TimeSeriesBenchmarkDataset(df, seq_len=96, pred_len=96, split="test", scaler=train_ds.scaler)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    # x: [batch_size, 96, n_features]
    # y: [batch_size, 96, n_features]
    break
```

---

## 📊 Zestawienie Benchmarków (LTSF)

Wszystkie zbiory znajdują się w `data/raw/`:

| Zbiór | Plik | Krok czasowy | Liczba wierszy | Liczba cech | Opis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ETTh1** | `ETTh1.csv` | 1 godzina | 17,420 | 7 | Transformator 1 (temp. oleju + obciążenia) |
| **ETTh2** | `ETTh2.csv` | 1 godzina | 17,420 | 7 | Transformator 2 (temp. oleju + obciążenia) |
| **ETTm1** | `ETTm1.csv` | 15 minut | 69,680 | 7 | Transformator 1 wysoka rozdzielczość |
| **ETTm2** | `ETTm2.csv` | 15 minut | 69,680 | 7 | Transformator 2 wysoka rozdzielczość |
| **Weather** | `weather.csv` | 10 minut | 52,696 | 21 | Pomiary stacji meteorologicznej Jena |
| **Electricity** | `electricity.csv` | 1 godzina | 26,304 | 321 | Zużycie prądu 321 klientów w Portugalii |
| **Traffic** | `traffic.csv` | 1 godzina | 17,544 | 862 | Obłożenie pasów autostradowych San Francisco |

---

## 📚 Baza Wiedzy i Artykuły Naukowe

W katalogu [**`docs/papers/`**](file:///Users/rzadkows/coding/studies/machine-learning/project/docs/papers) znajdują się pełne teksty artykułów źródłowych.  
Syntetyczne podsumowanie najważniejszych wniosków, matematycznych formulacji i argumentów dla każdego artykułu znajdziesz w dokumencie:
👉 [**`docs/PAPERS_SUMMARY.md`**](file:///Users/rzadkows/coding/studies/machine-learning/project/docs/PAPERS_SUMMARY.md)

1. **DLinear** (AAAI 2023): *"Are Transformers Effective for Time Series?"*
2. **PatchTST** (ICLR 2023): *"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"*
3. **iTransformer** (ICLR 2024): *"iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"*
4. **LinearTSModels** (2024): *"Are Linear Models Really Effective for Time Series Forecasting?"*
5. **CATS** (2024): *"CATS: Enhancing Multivariate Time Series Forecasting by Cross-Attention"*
6. **Forecast Evaluation** (2022): *"Evaluating Time Series Forecasting Models: An empirical study on performance estimation methods"*

---

## 🤝 Konwencje Pracy Zespołowej

1. **Nowe modele**: umieszczaj w `src/ts_forecasting_benchmark/models/nazwa_modelu.py` dziedzicząc z `torch.nn.Module`.
2. **Nowe notebooki**: umieszczaj w `notebooks/`, nazywając je prefiksem numerycznym, np. `01_eda_etth1.ipynb`, `02_dlinear_pilot.ipynb`.
3. **Nie wrzucaj wag do gita**: pliki `*.pt` i `*.ckpt` zapisuj w `results/checkpoints/` (są automatycznie ignorowane przez `.gitignore`).
4. **Zawsze używaj `set_seed(42)`**: to gwarantuje, że wyniki uzyskane na Twoim komputerze będą identyczne na komputerze reszty zespołu.
