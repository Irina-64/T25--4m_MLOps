# Лекция 10: Тестирование ML-моделей в Production
## Полный подробный конспект

---

## Слайд 1 — Введение: зачем тестировать ML-модели

### Контекст и вводная мотивация:

Тестирование — это не просто проверка кода. В ML-системах тестирование имеет особое значение, потому что:
- Модель может выглядеть как «черный ящик» 
- Баги в данных или логике проявляются не сразу
- Последствия ошибки могут быть дорогостоящими

**Реальные примеры дорогих ошибок:**

1. **Amazon Recruiting Tool (2018)**

Amazon создал ML-модель для отбора резюме. 
Модель оказалась предвзятой против женщин, 
так как тренировалась на исторических данных, 
где мужчин было больше. 
Это вызвало скандал, потери репутации и судебные проблемы.

   - ML-модель для отбора резюме смещена против женщин
   - Причина: тренировалась на исторических данных где мужчин было больше
   - Потери: репутация, судебные издержки
   - **Вывод:** нужны тесты на bias и fairness

2. **Microsoft Tay Chatbot (2016)**
Чат-бот Tay обучился на публичных твитах, в результате начал 
публиковать оскорбления и расистские высказывания. 
Его отключили через несколько часов после запуска. 
Причина — недостаточное тестирование и валидация данных.

   - Чат-бот обучился на Twitter и начал писать оскорбления
   - Причина: недостаточная валидация обучающих данных
   - Потери: отключение за часы
   - **Вывод:** нужны тесты на качество и безопасность данных

3. **Uber's Autonomous Vehicles (2018)**
Автомобиль Uber, оснащённый ML для автономного вождения, 
не распознал пешехода — произошла смертельная авария. 
Ошибка связана с неадекватным тестированием крайних сценариев 
и не обнаруженным багом в модели.

   - Машина не распознала пешехода → смертельный исход
   - Причина: недостаточное тестирование edge-cases
   - Потери: человеческая жизнь, расследование, потеря доверия
   - **Вывод:** нужны тесты на все возможные сценарии


4. **Apple Card Gender Bias (2019)**
ML модель кредитного скоринга Apple Card демонстрировала предвзятость
 против женщин, в результате чего женские заявки на кредит часто отклонялись 
или выдавалось меньше средств. 
Это вызвало громкий общественный резонанс и расследование.

5. **Credit Scoring Errors**
Во многих банках ML модели, если не тестировать тщательно, 
могут необоснованно отклонять заявки или выдавать кредиты 
с неправильным риском, что приводит к финансовым потерям 
и повышенным рискам невозврата.

Эти примеры подчёркивают важность тщательного тестирования моделей, 
мониторинга и проверки качества данных и предсказаний, 
чтобы избежать дорогостоящих последствий.

### Зачем нужно тестирование в ML:

1. **Обнаружение Data Issues** — пропуски, выбросы, ошибки в данных
2. **Проверка Quality** — убедиться модель работает как ожидается
3. **Предотвращение Regression** — новая версия не хуже старой
4. **Выявление Bias** — модель не дискриминирует подгруппы
5. **Обеспечение Safety** — модель безопасна в production
6. **Compliance** — соответствие регуляторным требованиям

### Интерактивный момент:

Опрос студентов:
- "Встречали ли вы баги в ML-коде? Как вы их обнаружили?"
- "Какие типы ошибок вас больше всего беспокоят?"
- "Как вы сейчас тестируете свои модели?"

---

## Слайд 2 — Пирамида тестирования в ML

### Структура (от базы к вершине):

```
                    ▲
                   / \
                  /   \  E2E Tests
                 /     \ (1-2)
                /-------\
               /         \ Integration Tests
              /           \ (5-10)
             /             \
            /---------------\
           /                 \ Unit Tests + Data Tests
          /                   \ (70-80)
         /____________________\
```

### Объяснение каждого уровня:

#### **Unit Tests & Data Tests (70-80% тестов)**

Самые быстрые и дешевые тесты. Проверяют отдельные компоненты.

**Unit Tests:**
- Функции для нормализации, оценки метрик
- Преобразование признаков
- Утилиты обработки данных

**Data Tests:**
- Проверка целостности данных
- Валидация схемы
- Поиск аномалий

**Примеры:**
```python
def test_normalize_handles_zero_variance():
    """Проверка что normalize не падает на одинаковые значения"""
    data = np.array([5, 5, 5, 5])
    result = normalize(data)
    assert np.isfinite(result).all()

def test_one_hot_encoding_unknown_category():
    """Проверка обработки неизвестной категории"""
    encoder = OneHotEncoder()
    encoder.fit(['a', 'b', 'c'])
    
    # Что происходит при неизвестной категории?
    with pytest.raises(ValueError):
        encoder.transform(['d'])
```

пример кода Python-теста test_preprocessing_handles_missing_values, 
который проверяет обработку пропущенных значений функцией 
препроцессинга, например, через заполнение средним:
```
import numpy as np
import pytest
from sklearn.impute import SimpleImputer

def preprocess_handle_missing(X):
    # Используем среднее для заполнения пропусков
    imputer = SimpleImputer(strategy="mean")
    return imputer.fit_transform(X)

def test_preprocessing_handles_missing_values():
    X = np.array([
        [1.0, 2.0, np.nan],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0]
    ])
    X_processed = preprocess_handle_missing(X)

    # Проверяем, что после обработки пропусков нет
    assert not np.isnan(X_processed).any()
    # Проверяем, что размерность не изменилась
    assert X_processed.shape == X.shape
    # Для NaN на месте [0,2] ожидаем (6+9)/2 = 7.5
    assert X_processed[0,2] == pytest.approx(7.5)
    # Для NaN на месте [1,1] ожидаем (2+8)/2 = 5.0
    assert X_processed[1,1] == pytest.approx(5.0)

```
этот тест:

* Создаёт матрицу с пропусками (np.nan)
* Запускает функцию препроцессинга
* Проверяет, что пропусков больше нет и размерности соответствуют
* Проверяет корректное заполнение конкретных позиций

#### **Integration Tests (5-10% тестов)**

Проверяют работу нескольких компонентов вместе.

**Примеры:**
- Pipeline: load data → preprocess → train → evaluate
- Проверка что output одного компонента подходит для input следующего
- Тесты на small subset данных для быстроты

```python
def test_full_pipeline_on_sample_data():
    """Интеграционный тест полного pipeline"""
    
    # Загружаем тестовые данные
    X, y = load_test_data(n_samples=100)
    
    # Запускаем pipeline
    pipeline = Pipeline([
        ('preprocessor', Preprocessor()),
        ('feature_selector', SelectKBest()),
        ('model', RandomForestClassifier())
    ])
    
    # Проверяем что pipeline работает
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    
    # Базовые проверки
    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()
    assert pipeline.score(X, y) > 0.5  # Baseline
```

#### **E2E Tests (1-2 тестов)**

Тесты сквозной работы от начала до конца. Самые медленные и дорогие.

**Примеры:**
- API тест: запрос → ответ → проверка
- Deployment тест: модель развернута и работает
- Smoke test: базовая проверка что система работает

```python
def test_api_predict_endpoint():
    """Тест API эндпоинта"""
    
    # Создаем клиент
    client = TestClient(app)
    
    # Отправляем запрос
    response = client.post("/predict", json={
        "carrier": "AA",
        "dep_hour": 9,
        "distance": 550.0
    })
    
    # Проверяем ответ
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert 0 <= data["prediction"] <= 1
```

---

## Слайд 3 — Тестирование данных до обучения

### Почему это критично:

**Принцип "Garbage In, Garbage Out" (GIGO):**
- Плохие данные → плохая модель (даже если логика идеальна)
- 80% времени в ML уходит на работу с данными
- Баги в данных обнаруживаются позже всего

### Типы проблем с данными:

#### **1. Отсутствующие значения (Missing Values)**

```python
# Проверка пропусков
df.isnull().sum()  # Какие столбцы имеют пропуски?
df.isnull().sum() / len(df)  # % пропусков в каждом столбце

# Правило: если > 50% пропусков → удалить столбец?
```

**Great Expectations тест:**
```python
import great_expectations as ge

df = ge.read_csv('data.csv')

# Ожидаем что age_hours имеет < 5% пропусков
df.expect_column_values_to_not_be_null(
    'age_hours',
    mostly=0.95  # Allow 5% nulls
)

# Ожидаем что price имеет 0 пропусков
df.expect_column_values_to_not_be_null('price')
```

#### **2. Некорректные значения (Invalid Values)**

```python
# age должен быть в [0, 120]
assert (df['age'] >= 0).all() and (df['age'] <= 120).all()

# price должен быть положительным
assert (df['price'] > 0).all()

# date должна быть в прошлом
assert (df['date'] <= datetime.now()).all()
```

**Pandera тест (более удобно):**
```python
import pandera as pa

# Определяем схему
schema = pa.DataFrameSchema({
    "age": pa.Column(int, checks=pa.Check.in_range(0, 120)),
    "price": pa.Column(float, checks=pa.Check.gt(0)),
    "carrier": pa.Column(str, checks=pa.Check.isin(['AA', 'UA', 'DL']))
})

# Валидируем данные
try:
    validated_df = schema.validate(df)
except pa.errors.SchemaError as e:
    print(f"Data validation failed: {e}")
```

#### **3. Выбросы (Outliers)**

```python
# IQR метод
Q1 = df['delay'].quantile(0.25)
Q3 = df['delay'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['delay'] < Q1 - 1.5*IQR) | (df['delay'] > Q3 + 1.5*IQR)]
print(f"Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
```

#### **4. Дублирующиеся записи (Duplicates)**

```python
# Проверка полных дубликатов
duplicates = df.duplicated()
print(f"Found {duplicates.sum()} duplicate rows")

# Проверка дубликатов по ключевым столбцам
duplicates_by_key = df.duplicated(subset=['user_id', 'date'])
print(f"Found {duplicates_by_key.sum()} duplicate transactions")
```

#### **5. Категориальные несоответствия (Categorical Mismatch)**

```python
# Какие категории в данных?
print(df['carrier'].unique())  # ['AA', 'UA', 'DL', 'SW', 'B6', 'unknown']

# Ожидали: ['AA', 'UA', 'DL', 'SW', 'B6']
# Есть неожиданное значение 'unknown' → проблема!

# Проверка
EXPECTED_CARRIERS = {'AA', 'UA', 'DL', 'SW', 'B6'}
actual_carriers = set(df['carrier'].unique())
unexpected = actual_carriers - EXPECTED_CARRIERS

if unexpected:
    print(f"Unexpected carriers: {unexpected}")
    raise ValueError(f"Unknown carriers: {unexpected}")
```

### Инструменты для Data Testing:

| Инструмент | Назначение | Пример |
|-----------|-----------|---------|
| **Great Expectations** | Full-featured data validation | `df.expect_column_values_between(...)` |
| **Pandera** | Schema validation | `schema.validate(df)` |
| **Pytest fixtures** | Тестовые данные | `@pytest.fixture` |
| **Deepchecks** | Автоматические checks | `DatasetValidation()` |

---

## Слайд 4 — Unit-тесты для функций предобработки

### Зачем нужны unit-тесты:

- Проверить что функция работает правильно на известных примерах
- Обнаружить баги рано, до попадания в production
- Облегчить рефакторинг (знаем что функция работает)
- Документация — тесты показывают как использовать функцию

### Примеры unit-тестов:

#### **Тест 1: Normalization (нормализация)**

```python
import numpy as np
from preprocessing import normalize

def test_normalize_scales_to_0_1():
    """Проверка что normalize масштабирует в [0, 1]"""
    data = np.array([0, 10, 20])
    result = normalize(data)
    
    expected = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_normalize_handles_single_value():
    """Проверка edge case: все значения одинаковые"""
    data = np.array([5, 5, 5, 5])
    result = normalize(data)
    
    # Обычно возвращает NaN или 0 (dependency от реализации)
    # Главное — не падает
    assert len(result) == len(data)

def test_normalize_handles_negative():
    """Проверка отрицательные значения"""
    data = np.array([-10, 0, 10])
    result = normalize(data)
    
    # Результат в [0, 1]
    assert result.min() >= 0
    assert result.max() <= 1
```

#### **Тест 2: One-Hot Encoding**

```python
from preprocessing import OneHotEncoder

def test_one_hot_encoding_basic():
    """Базовый тест one-hot encoding"""
    encoder = OneHotEncoder()
    
    # Обучаем на категориях
    encoder.fit(['red', 'green', 'blue'])
    
    # Трансформируем
    result = encoder.transform(['red', 'green'])
    
    # Проверяем форму
    assert result.shape == (2, 3)  # 2 записи, 3 категории
    
    # Проверяем что это one-hot (только нули и единицы)
    assert set(result.flatten()) == {0, 1}

def test_one_hot_encoding_unknown_category():
    """Проверка что неизвестная категория обрабатывается"""
    encoder = OneHotEncoder()
    encoder.fit(['red', 'green', 'blue'])
    
    # Что происходит при неизвестной категории?
    with pytest.raises(ValueError, match="Unknown category"):
        encoder.transform(['yellow'])
```

#### **Тест 3: Missing Value Imputation**

```python
from preprocessing import Imputer

def test_imputer_replaces_nan_with_mean():
    """Проверка что Imputer заполняет NaN средним значением"""
    data = np.array([[1, 2], [np.nan, 3], [5, 4]])
    
    imputer = Imputer(strategy='mean')
    result = imputer.fit_transform(data)
    
    # Нет NaN?
    assert not np.isnan(result).any()
    
    # Первый NaN заменен на среднее (1+5)/2 = 3
    expected_first_col = np.array([1.0, 3.0, 5.0])
    np.testing.assert_array_almost_equal(result[:, 0], expected_first_col)

def test_imputer_median_strategy():
    """Проверка median стратегия более устойчива к выбросам"""
    data = np.array([[1, 2], [np.nan, 3], [100, 4]])  # 100 — выброс
    
    imputer = Imputer(strategy='median')
    result = imputer.fit_transform(data)
    
    # Медиан из [1, 100] = 50.5 (более устойчив чем mean=50.5)
    assert result[1, 0] == 50.5  # Для этого примера медиан = (1+100)/2
```

#### **Использование pytest:**

```bash
# Запуск всех тестов
pytest tests/

# Запуск с verbose output
pytest tests/ -v

# Запуск конкретного теста
pytest tests/test_preprocessing.py::test_normalize_scales_to_0_1

# Запуск с coverage (что % тестировано?)
pytest tests/ --cov=preprocessing --cov-report=html
```

---

## Слайд 5 — Проверка корректности train/test split

### Почему это критично:

**Проблема: Data Leakage**
- Информация из test set просачивается в train set
- Модель выглядит лучше чем есть на самом деле
- В production модель работает хуже чем ожидалось

### Типы утечек данных:

#### **Утечка 1: Временная утечка (Temporal Leakage)**

```
НЕПРАВИЛЬНО:
──────────

Data: [Jan, Feb, Mar, Apr, May]
Split: 
  Train: [Jan, Feb, Mar, Apr, May]  ← Включены все!
  Test: [Jan, Feb, Mar, Apr, May]   ← То же самое!

Результат: модель "видела" тестовые данные при обучении


ПРАВИЛЬНО:
──────────

Data: [Jan, Feb, Mar, Apr, May]
Split:
  Train: [Jan, Feb, Mar]     ← История
  Test: [Apr, May]           ← Будущее
  
Результат: обучение на прошлом, тест на будущем
```

**Проверка:**
```python
import pandas as pd

# Убедитесь что test set временно ПОСЛЕ train set
train_dates = df[df.index.isin(train_indices)]['date']
test_dates = df[df.index.isin(test_indices)]['date']

assert train_dates.max() < test_dates.min(), "Temporal leakage detected!"
print(f"Train: {train_dates.min()} to {train_dates.max()}")
print(f"Test: {test_dates.min()} to {test_dates.max()}")
```

#### **Утечка 2: Feature Leakage (признак содержит target)**

```
НЕПРАВИЛЬНО:
──────────
Признаки:
  - passenger_class (класс билета)
  - delay_in_minutes  ← Это и есть TARGET! Утечка!
  
Модель "учится" что delay_in_minutes == target
В production нет этого признака → модель падает


ПРАВИЛЬНО:
──────────
Признаки:
  - passenger_class
  - dep_hour (час вылета)
  - distance (расстояние)
  
Target:
  - delay_in_minutes (то что хотим предсказать)
```

**Проверка:**
```python
# Проверьте корреляцию каждого признака с target
correlations = X.corrwith(y).sort_values(ascending=False)

# Если корреляция = 1.0 или близка → УТЕЧКА!
print(correlations)
for feature, corr in correlations.items():
    if abs(corr) > 0.99:
        print(f"WARNING: {feature} has perfect correlation with target!")
```

#### **Утечка 3: Set Leakage (дублирующиеся примеры)**

```
ПРОБЛЕМА:
────────
Один и тот же пример (с одинаковыми признаками) 
может быть в train И test set

Результат: модель видела этот пример при обучении
            поэтому может легко его предсказать

ПРОВЕРКА:
────────
# Проверяем что нет одинаковых строк в train и test
train_set = set(map(tuple, X_train.values))
test_set = set(map(tuple, X_test.values))

intersection = train_set & test_set
if intersection:
    print(f"WARNING: {len(intersection)} duplicate examples in train and test!")
```

### Проверка что распределения похожи:

```python
import matplotlib.pyplot as plt

# Проверяем что распределение признаков в train ~ test
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

features = ['age', 'income', 'delay']
for i, feature in enumerate(features):
    axes[i].hist(X_train[feature], alpha=0.5, label='Train')
    axes[i].hist(X_test[feature], alpha=0.5, label='Test')
    axes[i].legend()
    axes[i].set_title(feature)

plt.tight_layout()
plt.show()

# Статистический тест (KS test)
from scipy.stats import ks_2samp

for feature in features:
    statistic, p_value = ks_2samp(X_train[feature], X_test[feature])
    if p_value < 0.05:
        print(f"WARNING: {feature} has different distribution in train vs test!")
    else:
        print(f"OK: {feature} distributions are similar")
```

---

## Слайд 6 — Модульные тесты для модели

### Тесты предсказания:

#### **Тест 1: Output Shape**

```python
def test_model_predict_output_shape():
    """Проверка что модель возвращает правильную форму"""
    
    X_test = np.random.randn(10, 5)  # 10 примеров, 5 признаков
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Предсказание
    predictions = model.predict(X_test)
    
    # Проверяем форму
    assert predictions.shape == (10,), f"Expected shape (10,), got {predictions.shape}"

def test_model_predict_proba_shape():
    """Проверка shape вероятностей"""
    
    X_test = np.random.randn(10, 5)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Вероятности
    probas = model.predict_proba(X_test)
    
    # Проверяем форму
    assert probas.shape == (10, 2), f"Expected (10, 2), got {probas.shape}"
```

#### **Тест 2: Valid Output Values**

```python
def test_model_predictions_are_valid():
    """Проверка что предсказания в допустимом диапазоне"""
    
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Для классификации: только 0 или 1
    assert set(predictions) <= {0, 1}, "Invalid class predictions"
    
    # Вероятности: в [0, 1]
    probas = model.predict_proba(X_test)
    assert (probas >= 0).all() and (probas <= 1).all(), "Invalid probabilities"
    
    # Сумма вероятностей = 1
    assert np.allclose(probas.sum(axis=1), 1.0), "Probabilities don't sum to 1"
```

#### **Тест 3: Edge Cases**

```python
def test_model_handles_zero_variance_feature():
    """Проверка модель не падает когда признак не меняется"""
    
    X = np.array([[1, 1], [1, 2], [1, 3]])  # Первый столбец = 1 везде
    y = np.array([0, 1, 0])
    
    model = RandomForestClassifier()
    
    # Не должно падать
    try:
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)
    except Exception as e:
        pytest.fail(f"Model failed on zero-variance feature: {e}")

def test_model_handles_all_same_class():
    """Проверка когда все примеры одного класса"""
    
    X = np.random.randn(10, 5)
    y = np.zeros(10)  # Все класс 0
    
    model = RandomForestClassifier()
    
    try:
        model.fit(X, y)
        predictions = model.predict(X)
        # Должны быть все 0
        assert (predictions == 0).all()
    except Exception as e:
        pytest.fail(f"Model failed on single class: {e}")

def test_model_handles_nan_in_features():
    """Проверка модель не падает при NaN в признаках"""
    
    X = np.random.randn(10, 5)
    X[0, 0] = np.nan  # Добавляем NaN
    y = np.random.randint(0, 2, 10)
    
    # Модель с обработкой NaN
    model = Pipeline([
        ('imputer', SimpleImputer()),
        ('classifier', RandomForestClassifier())
    ])
    
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
```

---

## Слайд 7 — Тестирование метрик качества

### Почему это важно:

Нельзя просто вычислить accuracy/F1 и считать что все ОК. Нужно:
- Сравнить с baseline
- Проверить на кросс-валидации
- Убедиться нет переобучения

### Типы тестов:

#### **Тест 1: Baseline Comparison**

```python
def test_model_better_than_baseline():
    """Проверка модель лучше чем baseline"""
    
    # Baseline: всегда предсказываем самый частый класс
    baseline_pred = np.full_like(y_test, y_train.mode()[0])
    baseline_acc = accuracy_score(y_test, baseline_pred)
    
    # Наша модель
    model = train_model(X_train, y_train)
    model_pred = model.predict(X_test)
    model_acc = accuracy_score(y_test, model_pred)
    
    # Модель должна быть лучше baseline
    assert model_acc > baseline_acc, \
        f"Model accuracy {model_acc:.3f} not better than baseline {baseline_acc:.3f}"
```

#### **Тест 2: Cross-Validation Score**

```python
def test_model_stable_across_folds():
    """Проверка модель стабильна на разных folds"""
    
    from sklearn.model_selection import cross_val_score
    
    model = RandomForestClassifier()
    
    # Cross-validation с 5 folds
    scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    
    # Проверяем что scores не сильно варьируются
    mean_score = scores.mean()
    std_score = scores.std()
    
    print(f"CV Scores: {scores}")
    print(f"Mean: {mean_score:.3f}, Std: {std_score:.3f}")
    
    # Если std слишком большой → модель нестабильна
    assert std_score < 0.1, f"High variance in CV scores: std={std_score:.3f}"
    
    # Общее качество должно быть хорошим
    assert mean_score > 0.75, f"Low mean CV score: {mean_score:.3f}"
```

#### **Тест 3: Проверка Overfitting**

```python
def test_model_not_overfitting():
    """Проверка что модель не переобучена"""
    
    model = train_model(X_train, y_train)
    
    # Score на обучающих данных
    train_score = model.score(X_train, y_train)
    
    # Score на тестовых данных
    test_score = model.score(X_test, y_test)
    
    print(f"Train Score: {train_score:.3f}")
    print(f"Test Score: {test_score:.3f}")
    print(f"Difference: {train_score - test_score:.3f}")
    
    # Разница должна быть небольшой (< 5%)
    diff = train_score - test_score
    assert diff < 0.05, \
        f"Possible overfitting: train={train_score:.3f}, test={test_score:.3f}"
    
    # Test score должен быть хорошим
    assert test_score > 0.75, f"Poor test performance: {test_score:.3f}"
```

#### **Тест 4: Метрики для дисбалансированных данных**

```python
def test_model_on_imbalanced_data():
    """Проверка модель на дисбалансированных данных"""
    
    # Данные: 95% класс 0, 5% класс 1 (дисбалансировано)
    X, y = make_imbalanced_dataset(n_samples=1000, ratio=0.05)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Не используем accuracy (будет 95% даже если никогда не предсказывает класс 1)
    # Используем F1, Precision, Recall
    
    y_pred = model.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    print(f"F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}")
    
    # Для меньшего класса важен recall (поймали ли мы положительные примеры?)
    assert recall > 0.7, f"Low recall for minority class: {recall:.3f}"
```

---

## Слайд 8 — Проверка устойчивости (Robustness) модели

### Зачем нужны robustness тесты:

Модель может работать хорошо на чистых данных но падать на реальных данных с шумом, выбросами, ошибками.

### Типы robustness тестов:

#### **Тест 1: Добавление шума**

```python
def test_model_robust_to_noise():
    """Проверка модель работает с зашумленными данными"""
    
    # Чистые данные
    X_test_clean = X_test.copy()
    
    # Добавляем гауссовский шум
    noise = np.random.normal(0, 0.1, X_test_clean.shape)
    X_test_noisy = X_test_clean + noise
    
    model = train_model(X_train, y_train)
    
    # Предсказания
    pred_clean = model.predict_proba(X_test_clean)[:, 1]
    pred_noisy = model.predict_proba(X_test_noisy)[:, 1]
    
    # Предсказания должны быть похожи (не сильно отличаться)
    mean_diff = np.abs(pred_clean - pred_noisy).mean()
    print(f"Mean difference with noise: {mean_diff:.3f}")
    
    # Если шум добавляет много изменений → модель нестабильна
    assert mean_diff < 0.1, f"Model too sensitive to noise: {mean_diff:.3f}"
```

#### **Тест 2: Выбросы (Outliers)**

```python
def test_model_handles_outliers():
    """Проверка модель не падает на выбросы"""
    
    # Создаем выбросы (очень большие значения)
    X_test_with_outliers = X_test.copy()
    X_test_with_outliers[0] = X_test_with_outliers[0] * 1000  # Умножаем на 1000
    
    model = train_model(X_train, y_train)
    
    # Не должно падать
    try:
        predictions = model.predict(X_test_with_outliers)
        assert len(predictions) == len(X_test_with_outliers)
    except Exception as e:
        pytest.fail(f"Model failed on outliers: {e}")
    
    # Проверяем что предсказания остаются в допустимом диапазоне
    assert np.isfinite(predictions).all(), "Predictions contain NaN or Inf"
```

#### **Тест 3: Adversarial Examples**

```python
def test_model_on_adversarial_examples():
    """Проверка модель на adversarial примерах"""
    
    # Adversarial пример: пример специально создан для обмана модели
    from foolbox import PyTorchModel, FGSM
    
    model = train_model(X_train, y_train)
    
    # Создаем adversarial примеры
    attack = FGSM()
    adversarial_examples = attack(model, X_test, labels=y_test)
    
    # Проверяем что модель может их предсказать
    # (может быть с худшей accuracy, но все равно работает)
    try:
        predictions = model.predict(adversarial_examples)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy on adversarial examples: {accuracy:.3f}")
    except Exception as e:
        pytest.fail(f"Model failed on adversarial examples: {e}")
```

#### **Тест 4: Out-of-Distribution Examples**

```python
def test_model_on_out_of_distribution():
    """Проверка модель на данных вне распределения"""
    
    model = train_model(X_train, y_train)
    
    # Out-of-distribution: примеры очень отличаются от training data
    # Например, если train на возрасте 18-65, то OOD - это 100 лет
    
    X_ood = np.random.uniform(-100, 100, (10, X_train.shape[1]))  # Очень далеко от нормального
    
    # Модель не должна падать
    try:
        predictions = model.predict(X_ood)
        probas = model.predict_proba(X_ood)
        
        # Проверяем что вероятности в допустимом диапазоне
        assert (probas >= 0).all() and (probas <= 1).all()
    except Exception as e:
        pytest.fail(f"Model failed on OOD data: {e}")
```

---

## Слайд 9 — Интеграционные тесты: сквозной pipeline

### Что такое E2E тест:

E2E (End-to-End) тест проверяет всю цепочку:
data loading → preprocessing → model training → evaluation

### Пример полного E2E теста:

```python
def test_full_ml_pipeline():
    """Полный интеграционный тест ML pipeline"""
    
    # 1. Load данные
    X, y = load_test_dataset()
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # 3. Preprocessing
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 4. Обучение модели
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train_processed, y_train)
    
    # 5. Evaluation
    y_pred = model.predict(X_test_processed)
    
    # 6. Проверки
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    assert accuracy > 0.7, f"Accuracy too low: {accuracy:.3f}"
    assert f1 > 0.6, f"F1 too low: {f1:.3f}"
    
    print(f"Pipeline test passed! Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
```

### Тесты на small subset данных:

```python
@pytest.fixture
def small_dataset():
    """Fixture с маленьким dataset для быстрых тестов"""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_pipeline_on_small_data(small_dataset):
    """Быстрый тест на маленькие данные"""
    X, y = small_dataset
    
    # Pipeline should work on small data
    pipeline = build_ml_pipeline()
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    
    assert len(predictions) == len(y)
```

---

## Слайд 10 — Regression тесты (предотвращение деградации)

### Зачем нужны regression тесты:

Когда вы меняете код или обновляете модель, нужно убедиться что качество не упало.

### Пример:

```python
import json

# Сохраняем baseline метрики
BASELINE_METRICS = {
    'accuracy': 0.85,
    'f1': 0.82,
    'precision': 0.88,
    'recall': 0.78
}

def test_model_no_regression():
    """Проверка что качество не упало"""
    
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Вычисляем текущие метрики
    current_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    # Сравниваем с baseline
    for metric_name, baseline_value in BASELINE_METRICS.items():
        current_value = current_metrics[metric_name]
        
        # Допускаем падение на 1%
        min_acceptable = baseline_value * 0.99
        
        assert current_value >= min_acceptable, \
            f"{metric_name} regressed: {current_value:.3f} < {min_acceptable:.3f}"
    
    print("✓ No regression detected")
```

### Сохранение метрик:

```python
import json
from datetime import datetime

def save_metrics(metrics, model_version):
    """Сохраняем метрики для будущих regression тестов"""
    
    filename = f'metrics_{model_version}.json'
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'version': model_version,
        'metrics': metrics
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    print(f"Metrics saved to {filename}")

# После обучения новой модели
metrics = {
    'accuracy': 0.87,
    'f1': 0.85,
    'auc': 0.92
}

save_metrics(metrics, model_version='v2')
```

---

## Слайд 11 — ML специфические анти-паттерны

### Анти-паттерн 1: Data Leakage

**Проблема:** информация из тестового набора просачивается в обучение

**Пример:**
```python
# НЕПРАВИЛЬНО:
from sklearn.preprocessing import StandardScaler

# Масштабируем ВСЕ данные вместе
X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Проблема: scaler "видел" тестовые данные при fit!


# ПРАВИЛЬНО:
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit только на train
X_test_scaled = scaler.transform(X_test)  # Transform test
```

### Анти-паттерн 2: Overfitting на test set

**Проблема:** постоянно смотрим на test метрики и подгоняем модель

**Решение:** hold-out валидационный set
```python
# Разбиваем на 3 части
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Train: тренируем
# Val: подбираем гиперпараметры
# Test: финальная оценка (не трогаем!)
```

### Анти-паттерн 3: Недостаточное покрытие edge-cases

**Проблема:** модель работает на нормальных данных но падает на необычных

**Решение:** специальные тесты
```python
def test_edge_cases():
    # Все нули
    assert model.predict([[0, 0, 0]]) is not None
    
    # Все единицы
    assert model.predict([[1, 1, 1]]) is not None
    
    # NaN
    X_nan = X_test.copy()
    X_nan[0, 0] = np.nan
    assert model.predict(X_nan) is not None
```

---

## Слайд 12 — Использование CI/CD для ML

### Что это:

CI/CD = Continuous Integration / Continuous Deployment
- Каждый push → автоматически запускаются тесты
- Если тесты не прошли → код не мержится в main
- Если тесты прошли → код автоматически деплоится

### Пример: GitHub Actions workflow

```yaml
# .github/workflows/ml-tests.yml

name: ML Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run data tests
      run: pytest tests/test_data.py -v
    
    - name: Run unit tests
      run: pytest tests/test_preprocessing.py -v
    
    - name: Run model tests
      run: pytest tests/test_model.py -v
    
    - name: Run integration tests
      run: pytest tests/test_integration.py -v
    
    - name: Generate coverage
      run: pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

### Как это работает:

```
Developer pushes code
        ↓
GitHub Actions triggered
        ↓
├─ Run pytest
├─ Check code coverage
├─ Lint code (flake8)
├─ Type checking (mypy)
└─ Test coverage > 80%?
        ↓
    All pass?
   /        \
YES         NO
 ↓           ↓
Merge    Block merge
        ↓
Deploy   (show errors)
to prod
```

---

## Слайд 13 — Инструменты для ML-тестирования

### Обзор инструментов:

| Инструмент | Назначение | Пример |
|-----------|-----------|---------|
| **Pytest** | Фреймворк для тестирования | `pytest tests/ -v` |
| **Great Expectations** | Data validation | `df.expect_column_values_to_not_be_null(...)` |
| **Pandera** | Schema validation | `schema.validate(df)` |
| **Deepchecks** | Auto ML checks | `Check.data_integrity()` |
| **MLflow** | Model tracking + registry | `mlflow.log_metric('accuracy', 0.9)` |
| **Evidently** | Model monitoring | `DataDriftReport()` |

### Установка:

```bash
pip install pytest great-expectations pandera deepchecks mlflow evidently
```

### Быстрый пример:

```python
import great_expectations as ge
from sklearn.datasets import load_iris

# Загружаем данные
iris = load_iris()
df = ge.from_pandas(iris.data)

# Проверяем
df.expect_column_values_to_be_between('sepal length (cm)', 4, 8)
df.expect_column_count_to_equal(4)

# Результат
validation_result = df.validate()
print(validation_result)
```

---

## Слайд 14 — Чек-лист ML-тестирования

### Pre-Training:

- [ ] Data quality checks пройдены
- [ ] Нет пропусков > 50%
- [ ] Нет выбросов (или обработаны)
- [ ] Нет дублей
- [ ] Distribution train ~ test

### During Training:

- [ ] Unit tests для препроцессинга
- [ ] Model обучается без ошибок
- [ ] Нет NaN/Inf в весах
- [ ] Loss убывает (нет NaN)

### Before Deployment:

- [ ] Accuracy > baseline на 5%+
- [ ] Нет overfitting (train/test gap < 5%)
- [ ] F1, Recall проверены (не только accuracy)
- [ ] Cross-validation scores стабильны
- [ ] Integration test пройден
- [ ] E2E smoke test пройден

### In Production:

- [ ] Regression test кажется час
- [ ] Мониторинг data drift
- [ ] Мониторинг model performance
- [ ] Alert setup (email/slack)
- [ ] Rollback plan готов

---

## Слайд 15 — Ссылки и домашнее задание

### Обязательная литература:

- **Great Expectations**: https://greatexpectations.io/ (бесплатно!)
- **Deepchecks**: https://deepchecks.com/ (open-source версия свободна)
- **Pandera**: https://pandera.readthedocs.io/en/stable/
- **Pytest**: https://docs.pytest.org/en/stable/
- **MLflow**: https://mlflow.org/docs/latest/

### Книги:

- "Machine Learning Design Patterns" by Valliappa Lakshmanan, Sara Robinson, Michael Munn (O'Reilly, 2020)
  - Глава "Data Validation" и "Testing"
  
- "Reliable Machine Learning" by Kush Varshney (MIT Press, 2021)
  - Полная книга о reliability и тестировании

### Статьи:

- "ML Test Score: A Rubric for ML Production Readiness" (Google)
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f25b86243fa4c63a6147d9130abe287db00c.pdf

- "Best Practices for Testing ML Models" (MLOps.community)
  https://mlops.community/ml-test-guide/

### Домашнее задание:

**Задача 1: Написать Data Tests**
- Загрузите CSV файл
- Напишите 5 Great Expectations / Pandera checks
- Примеры: no nulls, values in range, schema validation

**Задача 2: Написать Unit Tests**
- Напишите функцию normalize(), log_transform(), etc
- Напишите 3+ unit tests используя pytest
- Протестируйте edge cases (zero, NaN, negative)

**Задача 3: Интеграционный тест**
- Постройте ML pipeline
- Напишите E2E тест который:
  - Загружает данные
  - Тренирует модель
  - Оценивает качество
  - Проверяет что все работает

**Задача 4 (продвинутая): CI/CD**
- Создайте GitHub Actions workflow
- Настройте чтобы тесты запускались при каждом push
- Блокируйте merge если тесты не прошли

---

**Конец конспекта лекции 10**  
Версия 1.0 | Ноябрь 2025
