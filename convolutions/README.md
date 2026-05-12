# Convolutional Models

В этой папке находятся эксперименты со сверточными нейронными сетями для fraud detection. Основная идея блока — анализировать последовательность транзакций клиента через одномерные свертки по времени.

## Содержимое

- `compare.ipynb` — сравнение нескольких CNN-архитектур;
- `meanmax_tuning.ipynb` — расширенный подбор гиперпараметров для `ConvMeanMax`;
- `attention_tuning.ipynb` — расширенный подбор гиперпараметров для `CNNAttention`.

## Данные

Ноутбуки используют подготовленные файлы из папки `data`:

- `../data/train_clean.csv`
- `../data/val_clean.csv`
- `../data/test_clean.csv`

Предобработка признаков выполняется централизованно в `data/cleanup.ipynb`. В CNN-ноутбуках признаки не пересчитываются заново: данные считываются, группируются по клиентам и превращаются в последовательности.

Размеры транзакционных выборок:

- `train_clean.csv`: **57 364** транзакции, **24** колонки;
- `val_clean.csv`: **19 162** транзакции, **24** колонки;
- `test_clean.csv`: **19 136** транзакций, **24** колонки.

Доля мошеннических транзакций:

- `train`: **0.20%**
- `val`: **0.21%**
- `test`: **0.20%**

## Последовательности

Модели обучаются на уровне клиентов. Для каждого клиента формируется последовательность длиной до **100** транзакций.

Перед сборкой последовательности транзакции внутри клиента сортируются по времени:

- `tx_elapsed_seconds`
- `tx_elapsed_days`
- `tx_month`
- `tx_day`
- `tx_hour`
- `tx_minute`

Если у клиента больше 100 транзакций, берутся последние 100 после сортировки. Если меньше 100, последовательность дополняется нулями в начале.

Размеры после группировки:

- `train`: **2 245** клиентов, форма `(2245, 100, 22)`;
- `val`: **748** клиентов, форма `(748, 100, 22)`;
- `test`: **749** клиентов, форма `(749, 100, 22)`.

Распределение fraud-клиентов:

- `train`: **32** fraud-клиента из 2 245;
- `val`: **11** fraud-клиентов из 748;
- `test`: **11** fraud-клиентов из 749.

## Признаки

В каждую транзакцию внутри последовательности входит **22 признака**:

- `Amount`
- `abs_amount`
- `log_abs_amount`
- `amount_sign`
- `is_negative_amount`
- `tx_month`
- `tx_day`
- `tx_hour`
- `tx_minute`
- `tx_dayofweek`
- `tx_is_weekend`
- `tx_elapsed_seconds`
- `tx_elapsed_days`
- `tx_hour_sin`
- `tx_hour_cos`
- `tx_dayofweek_sin`
- `tx_dayofweek_cos`
- `ProviderId_id`
- `ProductId_id`
- `ProductCategory_id`
- `ChannelId_id`
- `PricingStrategy_id`

`CustomerId` используется только для группировки транзакций, а целевая метка клиента считается как максимум `FraudResult` по его транзакциям.

## Архитектуры

Во всех моделях используется общая идея: последовательность клиента имеет форму `(100, 22)`, после чего признаки обрабатываются одномерными свертками по временной оси.

В сравнении рассмотрены:

- `ConvMean` — несколько сверточных блоков и mean pooling по времени;
- `ConvMax` — несколько сверточных блоков и max pooling;
- `ConvMeanMax` — объединение mean pooling и max pooling;
- `ResidualCNN` — сверточные residual-блоки;
- `DilatedCNN` — dilated-свертки для расширения временного контекста;
- `CNNAttention` — сверточные признаки с attention pooling.

## `compare.ipynb`

В ноутбуке сравниваются 6 CNN-архитектур:

- `ConvMean`
- `ConvMax`
- `ConvMeanMax`
- `ResidualCNN`
- `DilatedCNN`
- `CNNAttention`

Сводные результаты на test:

| Model | Params | ROC-AUC | PR-AUC | Precision | Recall | F1 | F0.5 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DilatedCNN | 50 242 | 0.9947 | 0.8045 | 0.7500 | 0.8182 | 0.7826 | 0.7627 | 735 | 3 | 2 | 9 |
| ResidualCNN | 112 642 | 0.9931 | 0.7864 | 0.7273 | 0.7273 | 0.7273 | 0.7273 | 735 | 3 | 3 | 8 |
| ConvMax | 33 666 | 0.9959 | 0.8031 | 0.6923 | 0.8182 | 0.7500 | 0.7143 | 734 | 4 | 2 | 9 |
| CNNAttention | 46 083 | 0.9925 | 0.8012 | 0.6923 | 0.8182 | 0.7500 | 0.7143 | 734 | 4 | 2 | 9 |
| ConvMeanMax | 37 762 | 0.9926 | 0.7763 | 0.6923 | 0.8182 | 0.7500 | 0.7143 | 734 | 4 | 2 | 9 |
| ConvMean | 33 666 | 0.9947 | 0.7645 | 0.6667 | 0.5455 | 0.6000 | 0.6383 | 735 | 3 | 5 | 6 |

Лучший результат по `F0.5` в сравнении показала `DilatedCNN`:

- `F0.5 = 0.7627`
- `Precision = 0.7500`
- `Recall = 0.8182`
- `FN = 2`
- `FP = 3`

## `attention_tuning.ipynb`

В ноутбуке выполняется Optuna-тюнинг модели `CNNAttention`.

Подбирались:

- `hidden_dim`
- `kernel_size`
- `num_layers`
- `dropout`
- `learning_rate`
- `weight_decay`
- `optimizer_type`
- `criterion_type`
- `scheduler_type`
- `batch_size`
- `clip_grad`
- параметры для `weighted_ce` и `focal loss`

Текущий запуск:

- `50 trials`
- лучший `F0.5` на validation: **0.9574**
- лучший порог на validation: **0.56**

Лучшие гиперпараметры:

- `hidden_dim = 128`
- `kernel_size = 7`
- `num_layers = 2`
- `dropout = 0.3`
- `learning_rate = 0.002699`
- `weight_decay = 0.000308`
- `optimizer_type = adam`
- `criterion_type = ce`
- `scheduler_type = step`
- `batch_size = 16`
- `clip_grad = 2.0`

Итоговые результаты на test:

- `TN = 736`
- `FP = 2`
- `FN = 2`
- `TP = 9`

Метрики:

- `F0.5 = 0.8182`
- `F1 = 0.8182`
- `Precision = 0.8182`
- `Recall = 0.8182`
- `PR-AUC = 0.7686`
- `ROC-AUC = 0.9329`
- `Accuracy = 0.9947`

## `meanmax_tuning.ipynb`

В ноутбуке выполняется Optuna-тюнинг модели `ConvMeanMax`.

MeanMax объединяет два представления последовательности:

- mean pooling — общий паттерн поведения клиента;
- max pooling — самые сильные локальные активации.

Текущий запуск:

- `50 trials`
- лучший `F0.5` на validation: **0.9574**
- лучший порог на validation: **0.72**

Лучшие гиперпараметры:

- `hidden_dim = 128`
- `kernel_size = 7`
- `num_layers = 4`
- `dropout = 0.2`
- `learning_rate = 0.002141`
- `weight_decay = 0.000005`
- `optimizer_type = adamw`
- `criterion_type = ce`
- `scheduler_type = step`
- `batch_size = 64`
- `clip_grad = 0.5`

Итоговые результаты на test:

- `TN = 737`
- `FP = 1`
- `FN = 4`
- `TP = 7`

Метрики:

- `F0.5 = 0.8140`
- `F1 = 0.7368`
- `Precision = 0.8750`
- `Recall = 0.6364`
- `PR-AUC = 0.7594`
- `ROC-AUC = 0.9628`
- `Accuracy = 0.9933`

## Сравнение лучших CNN-результатов

| Модель | F0.5 | Precision | Recall | PR-AUC | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention_tuning.ipynb` | 0.8182 | 0.8182 | 0.8182 | 0.7686 | 0.9329 | 2 | 2 |
| `meanmax_tuning.ipynb` | 0.8140 | 0.8750 | 0.6364 | 0.7594 | 0.9628 | 1 | 4 |
| `DilatedCNN` из `compare.ipynb` | 0.7627 | 0.7500 | 0.8182 | 0.8045 | 0.9947 | 3 | 2 |

## Вывод

После перехода на единый preprocessing и сортировку транзакций по времени CNN-модели обучаются на корректных клиентских последовательностях.

По текущим результатам:

- лучшая CNN-модель по `F0.5` — tuned `CNNAttention`;
- `ConvMeanMax` после тюнинга дает самую высокую precision среди CNN-моделей, но пропускает больше fraud-клиентов;
- `DilatedCNN` из comparison показывает сильный baseline без Optuna и остается хорошим кандидатом для дальнейшего тюнинга.

Важно учитывать малое число fraud-клиентов: на validation и test всего по **11** положительных клиентов. Поэтому различия в `FN` на 1-2 клиента сильно меняют метрики.
