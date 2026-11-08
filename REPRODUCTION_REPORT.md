# Воспроизведение Neural Induction of Finite-State Transducers

Дата проверки: 2026-11-06. Основной объект проверки — статья Ginn, Palmer,
Hulden (Findings of ACL 2026). Для морфологического склонения использованы
официальные данные SIGMORPHON 2020 Task 0.

## Что делает метод

1. CRPAlign выравнивает входную и выходную строки; epsilon-входы жадно
   объединяются с соседними выходами (`bpe` в коде проекта).
2. Однослойная Elman RNN с `tanh` учится предсказывать следующий выходной
   символ из текущего скрытого состояния и следующего входного символа.
3. Штраф на спектральную норму (`0.1`) подталкивает динамику RNN к компактным
   устойчивым областям.
4. Скрытые состояния собираются на train и синтетических комбинациях
   `lemma x feature-tags`, стандартизуются и кластеризуются k-means.
5. Конфликтующие переходы разделяются Linear SVM или logistic regression;
   недостижимые состояния удаляются, автомат минимизируется.
6. На устройстве остаётся только детерминированный невзвешенный FST — RNN для
   инференса не требуется.

## Проверенный эксперимент

Датасет `czn` (Zenzontepec Chatino): 1088 train, 154 dev, 305 test. Это один
из четырёх датасетов абляций статьи и один из 24 SIGMORPHON 2020 языков в
основной таблице.

- Окружение: локальный Miniconda, Python 3.11.18, PyTorch 2.4.1 CPU,
  NumPy 2.1.1.
- Обучение: один Elman RNN, 200 эпох, dimension 64, batch 128, dropout 0.1,
  AdamW lr 0.001, label smoothing 0.1, spectral penalty 0.1, seed 0.
- Извлечение: 32 комбинации: clusters `{50,100,150,200}` x classifier
  `{SVM, logistic}` x threshold `{None,2,5,10}`.
- Выбор конфигурации сделан по dev F1; test не использовался при выборе.
- Полный авторский протокол намного дороже: 50 neural runs и 100 extraction
  runs для этого датасета.

Результат честного выбора по dev: 64.26% exact-match accuracy на test против
66.60% в статье (разница 2.34 п.п.), 100% accepted inputs, CER 0.0263 и 362
состояния. Размер PyFoma pickle — 127,847 байт. Исследовательский Python
runtime обработал 305 test-примеров за 0.300 с, около 1,015 строк/с на CPU.
Это не embedded-бенчмарк: плоский C runtime должен измеряться отдельно.

В сетке есть конфигурации с 66.23% test accuracy и всего 52 состояния, но их
dev score ниже; выбирать их постфактум по test было бы утечкой, поэтому они не
объявлены основным результатом.

Численные результаты находятся в `results/final/czn-local-reproduction.json`,
а сериализованный автомат — в `results/final/czn-best.fst`.

## Что пришлось исправить для воспроизводимости

- README не указывает обязательный `--mode` и не описывает основной objective
  `transduction`.
- `src.sweep` жёстко привязан к W&B entity авторов и не имеет локального режима.
- Диапазон epochs в коде (`200,400,600`) не совпадает со статьёй
  (`200,600,1000`).
- CRPAlign использует POSIX-функции и не собирается MinGW без compatibility
  слоя; WSL на тестовой машине не стартовал из-за отключённой виртуализации.
- На CPU код ошибочно выбирал Apple `mps`; добавлен корректный fallback `cpu`.
- `torch.compile` нельзя вызывать на Windows; добавлена проверка платформы.
- `NamedTemporaryFile` оставался открытым при передаче W&B, что даёт access
  denied на Windows.
- state-splitting classifier не имел `random_state`; добавлен seed 0.
- сериализация большого PyFoma-графа превышала стандартный recursion limit.

Локальный runner: `python -m scripts.reproduce_local`. Он работает с W&B
offline, сохраняет все trials в JSON и экспортирует выбранный FST.

На Windows полный повтор запускается из корня репозитория так:

```powershell
conda env create -f environment.windows.yml
conda run -n fst-distillation x86_64-w64-mingw32-gcc -O3 -Wall -Wextra `
  -shared src/crpalign/align.c -o src/crpalign/libalign.so
conda run -n fst-distillation python -m src.run_alignment `
  data/inflection czn --features --iterations 100
conda run -n fst-distillation python -m scripts.reproduce_local
```

Файлы `czn.trn` и `czn.dev` взяты из
`DEVELOPMENT-LANGUAGES/oto-manguean`, а размеченный `czn.tst` — из
`GOLD-TEST` официального репозитория `sigmorphon2020/task0-data`.

## Какую модель пробовать следующей

Для inflection сначала стоит оставить Elman RNN: простая Markov-динамика —
часть причины, по которой FST хорошо извлекается. LSTM/GRU могут улучшить
нейросетевую точность, но усложняют извлечение и не гарантируют лучший FST.

Главный следующий эксперимент — state-regularized RNN (SR-RNN) или
vector-quantized Elman RNN: конечный набор прототипов состояния обучается
совместно с моделью. Это может убрать нестабильный post-hoc k-means и прямо
контролировать число состояний. Сравнивать нужно по Pareto-кривой
`test accuracy / states / serialized bytes / lookup latency`, а не только по
loss промежуточной нейросети.

Для G2P и historical normalization нужен правый контекст. Практичный вариант —
bidirectional RNN с извлечением bimachine (ветка уже начата в репозитории),
после чего bimachine конвертировать в обычный FST. Transformer имеет смысл как
teacher, но не как непосредственный runtime на малом устройстве.

## Runtime автомата

- Самый лёгкий embedded-вариант: экспортировать состояния и дуги в плоские
  C-массивы и выполнять один binary-search/индексированный переход на символ.
  Для детерминированного unweighted FST это минимальная зависимость и
  предсказуемая память.
- Foma: C library и `flookup`, удобен для NLP и поддерживает импорт/экспорт
  AT&T/OpenFst. Хороший первый промышленный runtime.
- OpenFst: C++, наиболее богатые алгоритмы и стандартный interchange, но для
  микроконтроллера может быть тяжелее необходимого.
- HFST optimized lookup: формат специально для быстрого lookup; разумен, если
  важна совместимость с морфологическими инструментами HFST.
- PyFoma оставить для исследования/конвертации, но не для конечного
  малоресурсного устройства: это pure Python runtime.

Rust crate `fst` не является прямой заменой: он рассчитан прежде всего на
ациклические byte-string set/map transducers с числовыми outputs, а здесь нужны
общие строковые выходы на переходах и возможные циклы.

## Источники

- Статья: https://aclanthology.org/2026.findings-acl.1411.pdf
- Код: https://github.com/michaelpginn/fst-distillation
- SIGMORPHON 2020 Task 0: https://github.com/sigmorphon2020/task0-data
- State-Regularized RNN: https://proceedings.mlr.press/v97/wang19j.html
- Foma: https://fomafst.github.io/
- HFST optimized lookup: https://github.com/hfst/hfst/wiki/OptimizedLookupFormat
- OpenFst: https://www.openfst.org/
