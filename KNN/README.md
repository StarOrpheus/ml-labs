### KNN + ядерное сглаживание Надарая Ватсона
* Метрики: Евклида, Манхэттена, Чебышева
* Ядра: uniform, triangular, quartic
* Фиксированное / плавающее окна
* Leave-one-out cross-validation

```
mkdir _build && cd _build && cmake -GNinja .. && ninja && cd .. && ./_build/ml_lab1
```
Построить график F-меры по выбранному кортежу (типу окна, ядру, метрике) можно например в `gnuplot`
```
gnuplot> plot "result.dat" using 1:2 with lines
```
