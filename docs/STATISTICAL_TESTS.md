# Testy statystyczne

Moduł `ts_forecasting_benchmark.evaluation.statistical_tests` porównuje jakość
prognoz. Funkcje są też eksportowane z `ts_forecasting_benchmark.evaluation`.
Nie są automatycznie uruchamiane przez pilota: jeden wynik MSE na model nie
wystarcza do tych analiz. Pełny benchmark (issue #4) ma dostarczyć predykcje i reszty.

## Porównanie dwóch modeli

`diebold_mariano_test(y_true, y_pred1, y_pred2, h=1, power=2)` zwraca słownik
`statistic`, `pvalue`. Wejścia muszą mieć te same daty i kolejność.
Każda obserwacja oznacza prognozę o tym samym wyprzedzeniu `h`, wykonaną z kolejnego
punktu startowego. Nie należy podawać 96 kolejnych kroków jednej prognozy jako
96 niezależnych prognoz o horyzoncie 96. Implementacja zakłada kolejne punkty
startowe oddalone o jeden krok czasu.

`power=2` porównuje kwadraty błędów, `power=1` błędy bezwzględne. Ujemna statystyka
oznacza mniejszą średnią stratę modelu 1. Test jest dwustronny, uwzględnia
autokowariancje do opóźnienia `h-1` z wagami Bartletta i poprawkę
Harveya–Leybourne’a–Newbolda. Obliczenia wykonuje pakiet
[dieboldmariano](https://pypi.org/project/dieboldmariano/); nasza funkcja sprawdza
dane wejściowe i zwraca wynik w formacie zgodnym z pozostałymi testami.
Wymaga więcej obserwacji niż wynosi `h`; stała niezerowa różnica strat daje wyjątek,
bo nie da się oszacować jej wariancji. Identyczne straty zwracają `(0, 1)`.

`wilcoxon_test(losses_model1, losses_model2)` przyjmuje **straty**, np. `abs(y - pred)`
albo `(y - pred) ** 2`, a nie reszty ze znakiem. Zwraca `statistic`, `pvalue`.
Porównywane jednostki muszą być sparowane i niezależne, a rozkład ich różnic
symetryczny pod hipotezą zerową. Nakładające się okna nie zapewniają niezależności.
Nie stosujemy automatycznego zaokrąglania różnic; precyzję trzeba ustalić zgodnie
z danymi, jeśli błędy numeryczne tworzą pozorne różnice rang.

## Porównanie wielu modeli i diagram CD

`friedman_nemenyi_test(scores)` przyjmuje DataFrame:

- wiersze: niezależne zadania ewaluacyjne, np. zbiory danych;
- kolumny: unikalne nazwy modeli;
- wartości: porównywalna miara straty, np. MSE; mniej oznacza lepiej.

Zwraca `statistic`, `pvalue` testu Friedmana, `mean_ranks` oraz macierz
`nemenyi_pvalues` z nazwami modeli. Remisy dostają średnią rangę. Przy całkowitym
remisie wynik to statystyka 0 i p-value 1. Przy małej liczbie zadań/modeli
przybliżenie chi-kwadrat Friedmana jest niedokładne. Horyzonty tego samego zbioru
mogą być zależne — nie należy traktować ich jako niezależnych powtórzeń bez uzasadnienia.

`plot_critical_difference(scores, alpha=0.05, ax=None)` przyjmuje tę samą macierz
strat, oblicza rangi i zwraca obiekt osi Matplotlib. Jeśli Friedman nie odrzuca
hipotezy zerowej, wszystkie modele są połączone. W przeciwnym razie połączenia
wynikają z testu Nemenyi. Połączenie oznacza brak wykrytej różnicy, nie dowód równoważności.

```python
from ts_forecasting_benchmark.evaluation import (
    friedman_nemenyi_test,
    plot_critical_difference,
)

# scores: DataFrame ze stratami wszystkich modeli na tych samych zadaniach.
result = friedman_nemenyi_test(scores)
print(result["pvalue"])
print(result["nemenyi_pvalues"])
ax = plot_critical_difference(scores)
ax.figure.savefig("results/figures/critical_difference.png", bbox_inches="tight")
```

Wszystkie testy odrzucają brakujące i nieskończone wartości zamiast usuwać je
po cichu. Niskie p-value jest przesłanką przeciw hipotezie zerowej przy spełnionych
założeniach; nie jest prawdopodobieństwem, że model jest najlepszy. Przy wielu
oddzielnych porównaniach par należy uwzględnić problem wielokrotnego testowania.

