# Transformers

В этой папке находятся эксперименты с transformer-моделями для fraud detection. Основная идея блока — анализировать последовательность транзакций клиента через self-attention, а не через рекуррентные слои.

## Содержимое

- `transformer_sequence_comparison.ipynb` — сравнение нескольких transformer-архитектур;
- `mean_tuning.ipynb` — подбор гиперпараметров для `Transformer Mean`;
- `meanmax_tuning.ipynb` — подбор гиперпараметров для `Transformer MeanMax`.

## Данные

Ноутбуки используют подготовленные файлы из папки `data`:

- `../data/train_clean.csv`
- `../data/val_clean.csv`
- `../data/test_clean.csv`

Предобработка признаков выполняется централизованно в `data/cleanup.ipynb`. В transformer-ноутбуках признаки не пересчитываются заново: данные только считываются, группируются по клиентам и превращаются в последовательности.

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

`CustomerId` используется только для группировки, а целевая метка клиента считается как максимум `FraudResult` по его транзакциям.

## Архитектуры

Во всех моделях используется общая transformer-основа:

- входная линейная проекция признаков в `d_model`;
- positional encoding;
- `TransformerEncoder`;
- dropout;
- classifier head.

Отличие моделей — в способе получить одно клиентское представление из последовательности:

- `TransformerMean` — усреднение hidden states по времени;
- `TransformerCLS` — специальный `CLS`-токен;
- `TransformerAttention` — attention pooling;
- `TransformerMeanMax` — конкатенация mean pooling и max pooling.

## `transformer_sequence_comparison.ipynb`

В ноутбуке сравниваются 4 архитектуры:

- `TransformerMean`
- `TransformerCLS`
- `TransformerAttention`
- `TransformerMeanMax`

Сводные результаты на test:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | F0.5 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransformerAttention | 0.9948 | 0.8612 | 0.8333 | 0.9091 | 0.8696 | 0.8475 | 736 | 2 | 1 | 10 |
| TransformerMean | 0.9979 | 0.8600 | 0.8333 | 0.9091 | 0.8696 | 0.8475 | 736 | 2 | 1 | 10 |
| TransformerCLS | 0.9966 | 0.8413 | 0.8000 | 0.7273 | 0.7619 | 0.7843 | 736 | 2 | 3 | 8 |
| TransformerMeanMax | 0.9978 | 0.8470 | 0.7143 | 0.9091 | 0.8000 | 0.7463 | 734 | 4 | 1 | 10 |

Лучшие модели в сравнении по `F0.5`:

- `TransformerAttention`: `F0.5 = 0.8475`
- `TransformerMean`: `F0.5 = 0.8475`

`TransformerAttention` чуть лучше по `PR-AUC`, а `TransformerMean` чуть лучше по `ROC-AUC`. Обе модели в этом сравнении пропустили только одного fraud-клиента.

## `mean_tuning.ipynb`

В ноутбуке выполняется Optuna-тюнинг модели `TransformerMean`.

Подбирались:

- `d_model`
- `nhead`
- `num_layers`
- `dim_feedforward`
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
- лучший `F0.5` на validation: **0.9302**
- лучший порог на validation: **0.80**

Лучшие гиперпараметры:

- `d_model = 192`
- `nhead = 8`
- `num_layers = 2`
- `dim_feedforward = 384`
- `dropout = 0.2`
- `learning_rate = 0.000923`
- `weight_decay = 0.000007`
- `optimizer_type = adamw`
- `criterion_type = weighted_ce`
- `scheduler_type = cosine`
- `batch_size = 16`
- `clip_grad = 1.5`
- `fraud_weight = 6.010428`

Итоговые результаты на test:

- `TN = 736`
- `FP = 2`
- `FN = 3`
- `TP = 8`

Метрики:

- `F0.5 = 0.7843`
- `F1 = 0.7619`
- `Precision = 0.8000`
- `Recall = 0.7273`
- `PR-AUC = 0.6794`
- `ROC-AUC = 0.9961`
- `Accuracy = 0.9933`

## `meanmax_tuning.ipynb`

В ноутбуке выполняется Optuna-тюнинг модели `TransformerMeanMax`.

MeanMax объединяет два представления последовательности:

- mean pooling — общий паттерн поведения клиента;
- max pooling — самые сильные локальные активации.

Текущий запуск:

- `50 trials`
- лучший `F0.5` на validation: **0.9302**
- лучший порог на validation: **0.84**

Лучшие гиперпараметры:

- `d_model = 192`
- `nhead = 8`
- `num_layers = 2`
- `dim_feedforward = 384`
- `dropout = 0.2`
- `learning_rate = 0.000923`
- `weight_decay = 0.000007`
- `optimizer_type = adamw`
- `criterion_type = weighted_ce`
- `scheduler_type = cosine`
- `batch_size = 16`
- `clip_grad = 1.5`
- `fraud_weight = 6.010428`

Итоговые результаты на test:

- `TN = 737`
- `FP = 1`
- `FN = 2`
- `TP = 9`

Метрики:

- `F0.5 = 0.8824`
- `F1 = 0.8571`
- `Precision = 0.9000`
- `Recall = 0.8182`
- `PR-AUC = 0.8473`
- `ROC-AUC = 0.9980`
- `Accuracy = 0.9960`

## Сравнение лучших transformer-результатов

| Модель | F0.5 | Precision | Recall | PR-AUC | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `meanmax_tuning.ipynb` | 0.8824 | 0.9000 | 0.8182 | 0.8473 | 0.9980 | 1 | 2 |
| `TransformerAttention` из comparison | 0.8475 | 0.8333 | 0.9091 | 0.8612 | 0.9948 | 2 | 1 |
| `TransformerMean` из comparison | 0.8475 | 0.8333 | 0.9091 | 0.8600 | 0.9979 | 2 | 1 |
| `mean_tuning.ipynb` | 0.7843 | 0.8000 | 0.7273 | 0.6794 | 0.9961 | 2 | 3 |

## Вывод

После перехода на единый preprocessing и сортировку транзакций по времени transformer-модели обучаются на корректных клиентских последовательностях.

По текущим результатам:

- лучшая transformer-модель по `F0.5` — tuned `TransformerMeanMax`;
- `TransformerAttention` и базовый `TransformerMean` из comparison дают чуть меньший `F0.5`, но лучше по recall и пропускают только одного fraud-клиента;
- tuned `TransformerMean` переоптимизировался под validation и на test оказался слабее comparison-вариантов.

Важно учитывать малое число fraud-клиентов: на validation и test всего по **11** положительных клиентов. Поэтому различия в `FN` на 1-2 клиента сильно меняют метрики.
