# 📚 Przegląd Literatury i Analiza Artykułów Źródłowych
> **Machine Learning Project 5**: *"Prognozowanie szeregów czasowych: czy mechanizm uwagi jest potrzebny?"*

Dokument zawiera szczegółową analizę 6 kluczowych publikacji naukowych zgromadzonych w katalogu [`docs/papers/`](file:///Users/rzadkows/coding/studies/machine-learning/project/docs/papers). Każdy artykuł został zanalizowany pod kątem jego wkładu w odpowiedź na pytanie: **kiedy i w jakiej formie mechanizm uwagi (attention) przynosi realną korzyść w prognozowaniu szeregów czasowych.**

---

## Spis Treści
1. [DLinear (AAAI 2023): Are Transformers Effective for Time Series?](#1-dlinear-aaai-2023)
2. [PatchTST (ICLR 2023): A Time Series is Worth 64 Words](#2-patchtst-iclr-2023)
3. [iTransformer (ICLR 2024): Inverted Transformers Are Effective](#3-itransformer-iclr-2024)
4. [LinearTSModels (ICML 2024): An Analysis of Linear Time Series Forecasting Models](#4-lineartsmodels-icml-2024)
5. [CATS (2024): Are Self-Attentions Effective for Time Series Forecasting?](#5-cats-2024)
6. [Forecast Evaluation (2022): Common Pitfalls and Best Practices](#6-forecast-evaluation-2022)
7. [Syntetyczna Macierz Porównawcza i Wnioski do Projektu](#7-syntetyczna-macierz-porównawcza)

---

## 1. DLinear (AAAI 2023)
**Pełny tytuł**: *Are Transformers Effective for Time Series Forecasting?*  
**Autorzy**: Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu (CUHK, IDEA)  
**Plik**: [`docs/papers/DLinear_2205.13504.pdf`](file:///Users/rzadkows/coding/studies/machine-learning/project/docs/papers/DLinear_2205.13504.pdf) | **ArXiv**: [2205.13504](https://arxiv.org/abs/2205.13504)

### Główna Teza i Problem Badawczy
Autorzy stawiają pod znakiem zapytania zasadność stosowania architektur Transformerowych (Informer, Autoformer, FEDformer) w długoterminowym prognozowaniu szeregów czasowych (LTSF).  
* **Problem uwagi**: Mechanizm Self-Attention z definicji jest **niezmienniczy na permutacje (permutation-invariant)**. Chociaż stosuje się kodowanie pozycyjne (positional encoding), dla ciągłych, uporządkowanych szeregów czasowych prowadzi to do niepotrzebnej utraty informacji o ścisłej kolejności czasowej.
* **Paradoks złożoności**: Złożone mechanizmy uwagi w dotychczasowych modelach nie uczyły się wcale relacji czasowych, lecz działały jak kosztowny filtr wygładzający.

### Architektura i Matematyka
Autorzy proponują rodzinę prostych, 1-warstwowych modeli liniowych (`LTSF-Linear`):

1. **`Linear`**:
   Pojedyncza warstwa liniowa bezpośrednio mapująca okno wejściowe $X \in \mathbb{R}^{L \times C}$ na horyzont $Y \in \mathbb{R}^{H \times C}$:
   $$\hat{Y} = X W$$
2. **`DLinear` (Decomposition Linear)**:
   Szereg wejściowy jest rozkładany za pomocą średniej ruchomej (Moving Average) na składową trendu $X_t$ i składową sezonową $X_s$:
   $$X_t = \text{AvgPool}(X), \quad X_s = X - X_t$$
   Każda składowa przechodzi przez niezależną warstwę liniową, po czym wyniki są sumowane:
   $$\hat{Y} = X_s W_s + X_t W_t$$
3. **`NLinear` (Normalization Linear)**:
   Radzi sobie z dryfem rozkładu (distribution shift) i niestacjonarnością. Odejmuje ostatnią znaną wartość sekwencji $X_L$, przepuszcza różnicę przez warstwę liniową i dodaje $X_L$ z powrotem:
   $$\hat{Y} = (X - X_L) W + X_L$$

### Kluczowe Wyniki Eksperymentalne
* Na 9 benchmarkach (w tym ETTh1/2, ETTm1/2, Weather, Electricity, Traffic) `DLinear` i `NLinear` pobiły Informera, Autoformera i FEDformera w niemal wszystkich scenariuszach, redukując błąd MSE średnio o **20%–50%**.
* Modele liniowe trenują się **10–100x szybciej** i zużywają ułamek pamięci RAM/VRAM (kilkadziesiąt parametrów zamiast milionów).
* **Zjawisko Lookback Horizon**: W Transformerach zwiększenie okna wejściowego $L$ z 96 do 720 często *pogarszało* wyniki (model przeuczał na szumie). W DLinear błąd malał wraz ze wzrostem $L$.

### Rola w Naszym Projekcie
DLinear i NLinear stanowią nasz **główny punkt odniesienia (strong baseline)**. Projekt ma sprawdzić, czy nowoczesne Transformery są w stanie pobić ten prosty model.

---

## 2. PatchTST (ICLR 2023)
**Pełny tytuł**: *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*  
**Autorzy**: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam (Princeton, IBM Research)  
**Plik**: [`docs/papers/PatchTST_2211.14730.pdf`](file:///Users/rzadkows/coding/studies/machine-learning/project/docs/papers/PatchTST_2211.14730.pdf) | **ArXiv**: [2211.14730](https://arxiv.org/abs/2211.14730)

### Odpowiedź na Krytykę DLinear
PatchTST to bezpośrednia odpowiedź obozu Transformerów na zarzuty Zeng et al. Autorzy wykazują, że wcześniejsze Transformery zawodziły nie z powodu samego mechanizmu uwagi, ale przez **błędną reprezentację tokenów (point-wise tokenization)**:
1. Traktowanie pojedynczego punktu w czasie $x_t$ jako tokena nie ma sensu semantycznego (pojedyncza liczba nie niesie informacji o trendzie czy dynamice).
2. Długość sekwencji $L$ powodowała, że macierz uwagi $O(L^2)$ pożerała pamięć, uniemożliwiając patrzenie w odległą przeszłość.
3. Mieszanie wielu kanałów (variates) w jednym kroku czasowym prowadziło do szumu korelacyjnego.

### Kluczowe Innowacje Architektoniczne
1. **Patching (Podział na pod-szeregi)**:
   Szereg o długości $L$ dzielony jest na nakładające się okna o długości $P$ (np. $P=16$) z krokiem $S$ (np. $S=8$).
   * Liczba tokenów spada z $L$ do $N = \lfloor (L - P) / S \rfloor + 2$.
   * Złożoność uwagi spada z $O(L^2)$ do $O(N^2)$ (quadratically reduced compute).
   * Każdy token reprezentuje lokalny kształt/wzorzec szeregu, a nie pojedynczą liczbę.
2. **Channel-Independence (CI - Niezależność Kanałów)**:
   Zamiast przetwarzać wszystkie zmienne wielowymiarowe naraz, model traktuje każdą zmienną jako osobny szereg jednowymiarowy. Wszystkie kanały przechodzą przez **ten sam, współdzielony enkoder Transformera**.

### Wyniki i Wnioski
* PatchTST **zdecydowanie pokonuje DLinear** we wszystkich zadaniach długoterminowych, szczególnie gdy okno wejściowe jest długie ($L = 336$ lub $L = 512$).
* Potwierdza, że mechanizm uwagi *ma sens*, pod warunkiem że operuje na poziomie semantycznych wycinków (patchy), a nie pojedynczych punktów.

### Rola w Naszym Projekcie
PatchTST to nasz **główny reprezentant nowoczesnego Transformera czasowego**. Będziemy porównywać jego jakość i narzut obliczeniowy z DLinear.

---

## 3. iTransformer (ICLR 2024)
**Pełny tytuł**: *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*  
**Autorzy**: Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long (Tsinghua University, Ant Group)  
**Plik**: [`docs/papers/iTransformer_2310.06625.pdf`](file:///Users/rzadkows/coding/studies/machine-learning/project/docs/papers/iTransformer_2310.06625.pdf) | **ArXiv**: [2310.06625](https://arxiv.org/abs/2310.06625)

### Nowatorska Koncepcja: Odwrócenie Osi Uwagi (Inverted Attention)
Tradycyjne Transformery tworzą token z wektora wszystkich zmiennych w danej chwili $t$, po czym liczą uwagę w czasie. iTransformer stawia sprawę na głowie:
* **Token = cała historia pojedynczej zmiennej**: Dla zmiennej $i$, wektor historii $[x_{1,i}, x_{2,i}, \dots, x_{L,i}] \in \mathbb{R}^L$ jest rzutowany przez MLP na wektor embeddingu o wymiarze $D$.
* **Mechanizm uwagi operuje MIĘDZY ZMIENNYMI (cross-variate)**:
  Macierz uwagi ma wymiary $C \times C$ (gdzie $C$ to liczba cech/kanałów), a nie $L \times L$.
* **Sieć Feed-Forward (FFN) modeluje czas**:
  Warstwy gęste wewnątrz bloku Transformera uczą się nieliniowych wzorców czasowych dla każdego kanału osobno.

### Dlaczego to Działa?
1. W szeregach wielowymiarowych (np. `Electricity` $C=321$, `Traffic` $C=862$) relacje między różnymi czujnikami są kluczowe, a punkty pomiarowe często mają opóźnienia fazowe.
2. Uwaga między zmiennymi jest **niezmiennicza na permutacje cech**, co idealnie pasuje do natury wielu zmiennych (kolejność kolumn w tabeli nie ma znaczenia!).
3. Złożoność czasowa nie zależy kwadratowo od długości okna $L$, więc model może bez problemu przyjmować $L=720$ lub $L=1000$.

### Rola w Naszym Projekcie
iTransformer odpowiada na pytanie: *"Gdzie w szeregach czasowych uwaga jest najbardziej potrzebna?"* Odpowiedź iTransformera: **nie wzdłuż osi czasu, lecz wzdłuż osi zmiennych**.

---

## 4. LinearTSModels (ICML 2024)
**Pełny tytuł**: *An Analysis of Linear Time Series Forecasting Models*  
**Autorzy**: William Toner, Luke Darlow (Edinburgh)  
**Plik**: [`docs/papers/LinearTSModels_2403.14587.pdf`](file:///Users/rzadkows/coding/studies/machine-learning/project/docs/papers/LinearTSModels_2403.14587.pdf) | **ArXiv**: [2403.14587](https://arxiv.org/abs/2403.14587)

### Matematyczna Demistyfikacja Modeli Liniowych
Autorzy przeprowadzają rygorystyczną analizę przestrzeni funkcyjnej modeli liniowych (`Linear`, `DLinear`, `NLinear`, `RLinear`).  
* **Dowód równoważności**: Wykazują, że większość wariantów normalizacji i dekompozycji w modelach liniowych sprowadza się matematycznie do **zwykłej regresji liniowej na rozszerzonym zbiorze cech (augmented feature space)** (np. z dodaną stałą lub kolumną ostatniej wartości).
* **Rozwiązanie analityczne (Closed-form OLS)**: Skoro te modele to w istocie klasyczna regresja liniowa z funkcją straty MSE, nie trzeba ich wcale trenować gradientowo przez setki epok!
  Rozwiązanie optymalne można wyznaczyć bezpośrednio wzorem:
  $$\hat{W} = (X^T X + \lambda I)^{-1} X^T Y$$
* **Wynik eksperymentu**: Analityczne rozwiązanie regresji grzbietowej (Ridge) osiąga lepsze lub identyczne wyniki w **72% przypadków testowych** w czasie ułamka sekundy.

### Rola w Naszym Projekcie
Przypomina zespołowi o krytycznym myśleniu: przed zachwytem nad skomplikowanymi sieciami neuronowymi należy upewnić się, czy optymalne rozwiązanie analityczne nie rozwiązuje problemu od ręki.

---

## 5. CATS (2024)
**Pełny tytuł**: *Are Self-Attentions Effective for Time Series Forecasting?*  
*(Cross-Attention-only Time Series transformer)*  
**Autorzy**: Dongbin Kim, Jinseong Park, Jaewook Lee, Hoki Kim (Seoul National University, Chung-Ang University)  
**Plik**: [`docs/papers/CATS_2405.16877.pdf`](file:///Users/rzadkows/coding/studies/machine-learning/project/docs/papers/CATS_2405.16877.pdf) | **ArXiv**: [2405.16877](https://arxiv.org/abs/2405.16877)

### Kluczowa Dychotomia: Self-Attention vs. Cross-Attention
Artykuł precyzyjnie rozdziela dwa pojęcia, które często wrzuca się do jednego worka:
1. **Self-Attention (Samo-uwaga)**: Porównywanie punktów historycznych między sobą ($X \leftrightarrow X$).
2. **Cross-Attention (Uwaga krzyżowa)**: Zapytanie z przyszłości do przeszłości ($Q_{\text{future}} \leftrightarrow K_{\text{past}}$).

### Główna Teza i Architektura
* Autorzy dowodzą, że **Self-Attention w szeregach czasowych jest zbędne i wprowadza szum**, ponieważ punkty historyczne są już lokalnie skorelowane.
* Proponują model **CATS**, który całkowicie usuwa bloki Self-Attention. Zamiast tego wykorzystuje wyłącznie **Cross-Attention**:
  * Zapytania ($Query$) są generowane przez wyuczone parametry zależne od horyzontu przyszłości ($H$).
  * Klucze ($Key$) i wartości ($Value$) pochodzą z przetworzonej historii ($L$).
* Model osiąga znakomite wyniki, zużywając znacznie mniej pamięci i parametrów niż pełny Transformer.

### Rola w Naszym Projekcie
Daje precyzyjną teoretyczną odpowiedź: **jeśli uwaga jest potrzebna, to przede wszystkim jako Cross-Attention łączące cel prognozy z kontekstem, a nie Self-Attention wewnątrz historii.**

---

## 6. Forecast Evaluation (2022)
**Pełny tytuł**: *Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices*  
**Autorzy**: Hansika Hewamalage, Klaus Ackermann, Christoph Bergmeir (Monash University, RMIT)  
**Plik**: [`docs/papers/ForecastEvaluation_2203.10716.pdf`](file:///Users/rzadkows/coding/studies/machine-learning/project/docs/papers/ForecastEvaluation_2203.10716.pdf) | **ArXiv**: [2203.10716](https://arxiv.org/abs/2203.10716)

### Metodologiczny Przewodnik dla Naszego Benchmarku
Ten artykuł (napisany przez twórców archiwum Monash) to "złoty standard" poprawnej metodologii badawczej w prognozowaniu szeregów czasowych. Przestrzega przed najczęstszymi błędami popełnianymi przez inżynierów ML:

### Główne Pułapki i Zasady:
1. **Wyciek danych (Data Leakage)**:
   * **BŁĄD**: Normalizacja danych (np. `StandardScaler`) dopasowana do całego zbioru przed podziałem na train/test.
   * **POPRAWNIE**: Skaler musi być wyliczony *wyłącznie* na zbiorze treningowym (tak jak zaimplementowaliśmy w [`dataset.py`](file:///Users/rzadkows/coding/studies/machine-learning/project/src/ts_forecasting_benchmark/data/dataset.py)).
2. **Zakaz losowego podziału (No Random K-Fold)**:
   * W szeregach czasowych nie wolno losować próbek walidacyjnych – narusza to strukturę przyczynowo-skutkową (autokorelację). Należy stosować wyłącznie podział chronologiczny lub Rolling-Origin Evaluation.
3. **Wybór metryk**:
   * MSE jest wrażliwe na wartości odstające, MAE mierzy błąd mediany.
   * W raportach należy podawać błędy zarówno w skali znormalizowanej, jak i oryginalnej.
4. **Weryfikacja statystyczna**:
   * Różnica MSE rzędu 0.005 może być dziełem przypadku losowego ziarna. Należy raportować wyniki dla kilku seedów i badać istotność różnic (np. test Wilcoxona).

---

## 7. Syntetyczna Macierz Porównawcza

Poniższa tabela podsumowuje, jak analizowane artykuły odpowiadają na główne pytanie projektu:

| Model / Artykuł | Czy uwaga jest potrzebna? | Gdzie leży uwaga? | Złożoność obliczeniowa | Największa zaleta | Słaby punkt |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DLinear / NLinear** *(Zeng 2023)* | ❌ **NIE** | Brak uwagi (1 warstwa liniowa + dekompozycja) | $O(L)$ (błyskawiczna) | Niezwykle szybki, mało parametrów, odporny na overfitting | Trudniej modeluje złożone korelacje nieliniowe |
| **Linear Models OLS** *(Toner 2024)* | ❌ **NIE** | Brak (zamknięte równanie analityczne Ridge) | $O(L^3)$ jednorazowo | Zero epok treningu, optymalne rozwiązanie w sekundę | Trudno skalować do olbrzymich macierzy cech |
| **PatchTST** *(Nie 2023)* |  **TAK** (warunkowo) | W czasie, ale **na poziomie patchy** ($N = L/S$) | $O((L/S)^2)$ | Świetna jakość dla długich $L \ge 336$, modeluje dynamikę podszeregów | Brak jawnego modelowania korelacji międzykanałowych (CI) |
| **iTransformer** *(Liu 2024)* |  **TAK** (w innej osi) | **Między zmiennymi** (cross-variate), nie w czasie | $O(C^2 + L)$ | Znakomity dla wielu zmiennych (np. Traffic $C=862$) | Słabszy, gdy liczba zmiennych jest mała (np. 1-7) |
| **CATS** *(Kim 2024)* | ⚖️ **Tylko Cross-Attn** | **Między celem $H$ a historią $L$** (zero Self-Attn) | $O(H \times L)$ | Brak szumu z self-attention, mniej parametrów | Mniej uniwersalny poza zadaniem LTSF |

---

## 8. Wnioski Końcowe dla Zespołu (Key Takeaways)

1. **Odpowiedź brzmi: "To zależy od osi i reprezentacji tokenów"**:
   * Proste mechanizmy uwagi punkt po punkcie w czasie są gorsze niż DLinear.
   * Mechanizm uwagi wygrywa z modelami liniowymi tylko wtedy, gdy:
     * Operuje na **patchach** zamiast punktów (PatchTST dla długiego kontekstu $L$).
     * Albo operuje **między zmiennymi** w systemach wielowymiarowych (iTransformer).
2. **Nasz plan eksperymentalny jest kompletny**:
   Mamy reprezentanta czystych modeli liniowych (`DLinear`/`NLinear`), Transformera patchowego (`PatchTST`) oraz Transformera odwróconego (`iTransformer`). Wyniki z tych trzech rodzin pozwolą zespołowi udzielić wyczerpującej, wieloaspektowej odpowiedzi na obronie projektu.

