# Recurrent Models

В этой папке находятся эксперименты с рекуррентными нейронными сетями для fraud detection. Основная идея блока — анализировать не отдельные транзакции, а последовательность операций каждого клиента.

## Содержимое

- `compare.ipynb` — сравнение нескольких recurrent-архитектур;
- `gru_tuning.ipynb` — расширенный подбор гиперпараметров для GRU;
- `bilstm_tuning.ipynb` — расширенный подбор гиперпараметров для BiLSTM.

## Данные

Ноутбуки используют уже подготовленные файлы из папки `data`:

- `../data/train_clean.csv`
- `../data/val_clean.csv`
- `../data/test_clean.csv`

Предобработка признаков выполняется централизованно в `data/cleanup.ipynb`. В recurrent-ноутбуках признаки не пересчитываются заново: данные только считываются, после чего группируются в клиентские последовательности.

Размеры исходных транзакционных выборок:

- `train_clean.csv`: **57 364** транзакции, **24** колонки;
- `val_clean.csv`: **19 162** транзакции, **24** колонки;
- `test_clean.csv`: **19 136** транзакций, **24** колонки.

Доля мошеннических транзакций:

- `train`: **0.20%**
- `val`: **0.21%**
- `test`: **0.20%**

## Последовательности

Модели обучаются на уровне клиентов. Для каждого клиента формируется последовательность транзакций длиной до **100** операций.

Важно: перед формированием последовательности транзакции внутри клиента сортируются по времени:

- `tx_elapsed_seconds`
- `tx_elapsed_days`
- `tx_month`
- `tx_day`
- `tx_hour`
- `tx_minute`

Если у клиента больше 100 транзакций, берутся последние 100 после сортировки. Если меньше 100, последовательность дополняется нулями в начале.

После группировки получаются такие размеры:

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

`CustomerId` используется только для группировки транзакций, а `FraudResult` — как целевая метка клиента через максимум по его транзакциям.

## `compare.ipynb`

В ноутбуке сравниваются 6 моделей:

- `RNN`
- `LSTM`
- `BiLSTM`
- `GRU`
- `ImprovedGRU`
- `ImprovedLSTM`

Сводные результаты на test:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | F0.5 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU | 0.9453 | 0.7260 | 0.7500 | 0.8182 | 0.7826 | 0.7627 | 735 | 3 | 2 | 9 |
| ImprovedLSTM | 0.9957 | 0.7163 | 0.7143 | 0.9091 | 0.8000 | 0.7463 | 734 | 4 | 1 | 10 |
| LSTM | 0.9757 | 0.7290 | 0.7000 | 0.6364 | 0.6667 | 0.6863 | 735 | 3 | 4 | 7 |
| BiLSTM | 0.9917 | 0.6250 | 0.6429 | 0.8182 | 0.7200 | 0.6716 | 733 | 5 | 2 | 9 |
| RNN | 0.9444 | 0.7299 | 0.6667 | 0.5455 | 0.6000 | 0.6383 | 735 | 3 | 5 | 6 |
| ImprovedGRU | 0.9962 | 0.7533 | 0.6667 | 0.3636 | 0.4706 | 0.5714 | 736 | 2 | 7 | 4 |

Лучший результат по `F0.5` в сравнении показала обычная `GRU`:

- `F0.5 = 0.7627`
- `Precision = 0.7500`
- `Recall = 0.8182`
- `FN = 2`
- `FP = 3`

`ImprovedLSTM` дала более высокий recall и пропустила только одного fraud-клиента, но по `F0.5` немного уступила `GRU`.

## `bilstm_tuning.ipynb`

В ноутбуке выполняется Optuna-тюнинг bidirectional LSTM-модели с attention/mean/max pooling и усиленной classifier head.

Подбирались:

- `hidden_dim`
- `num_layers`
- `dropout`
- `learning_rate`
- `weight_decay`
- `optimizer_type`
- `criterion_type`
- `scheduler_type`
- `batch_size`
- параметры для `weighted_ce` и `focal loss`

Текущий запуск:

- `50 trials`
- лучший `F0.5` на validation: **0.9091**
- лучший порог на validation: **0.61**

Лучшие гиперпараметры:

- `hidden_dim = 48`
- `num_layers = 2`
- `dropout = 0.5`
- `learning_rate = 0.002633`
- `weight_decay = 0.000001`
- `optimizer_type = adam`
- `criterion_type = focal`
- `scheduler_type = cosine`
- `batch_size = 64`
- `alpha_fraud = 6.532081`
- `gamma = 2.491746`

Итоговые результаты на test:

- `TN = 734`
- `FP = 4`
- `FN = 1`
- `TP = 10`

Метрики:

- `F0.5 = 0.7463`
- `F1 = 0.8000`
- `Precision = 0.7143`
- `Recall = 0.9091`
- `PR-AUC = 0.7482`
- `ROC-AUC = 0.9744`
- `Accuracy = 0.9933`

## `gru_tuning.ipynb`

В ноутбуке выполняется Optuna-тюнинг bidirectional GRU-модели с attention/mean/max pooling и усиленной classifier head.

Дополнительно для GRU подбирается `clip_grad`, потому что ограничение нормы градиента помогает стабилизировать обучение recurrent-сетей.

Текущий запуск:

- `50 trials`
- лучший `F0.5` на validation: **0.8974**
- лучший порог на validation: **0.87**

Лучшие гиперпараметры:

- `hidden_dim = 96`
- `num_layers = 2`
- `dropout = 0.1`
- `learning_rate = 0.001833`
- `weight_decay = 0.000598`
- `optimizer_type = adamw`
- `criterion_type = focal`
- `scheduler_type = step`
- `batch_size = 128`
- `clip_grad = 1.5`
- `alpha_fraud = 12.563881`
- `gamma = 1.042106`

Итоговые результаты на test:

- `TN = 735`
- `FP = 3`
- `FN = 4`
- `TP = 7`

Метрики:

- `F0.5 = 0.6863`
- `F1 = 0.6667`
- `Precision = 0.7000`
- `Recall = 0.6364`
- `PR-AUC = 0.7489`
- `ROC-AUC = 0.9922`
- `Accuracy = 0.9907`

## Сравнение лучших recurrent-результатов

| Модель | F0.5 | Precision | Recall | PR-AUC | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GRU` из `compare.ipynb` | 0.7627 | 0.7500 | 0.8182 | 0.7260 | 0.9453 | 3 | 2 |
| `bilstm_tuning.ipynb` | 0.7463 | 0.7143 | 0.9091 | 0.7482 | 0.9744 | 4 | 1 |
| `gru_tuning.ipynb` | 0.6863 | 0.7000 | 0.6364 | 0.7489 | 0.9922 | 3 | 4 |

## Вывод

После перехода на единый preprocessing и сортировку транзакций по времени recurrent-модели обучаются на корректных клиентских последовательностях.

По текущим результатам:

- лучшая recurrent-модель по `F0.5` — обычная `GRU` из `compare.ipynb`;
- tuned `BiLSTM` немного уступает по `F0.5`, но лучше ловит fraud-клиентов: `FN = 1`;
- tuned `GRU` переоптимизировалась под validation и на test оказалась слабее двух других вариантов.

Важно учитывать, что fraud-клиентов мало: на validation и test всего по **11** положительных клиентов. Поэтому результаты чувствительны к split и выбранному порогу.
