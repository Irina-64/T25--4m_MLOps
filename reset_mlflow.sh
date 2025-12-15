#!/bin/bash
# reset_mlflow.sh - очистка MLflow для чистого запуска

echo "🧹 Очистка MLflow для чистого запуска..."

# Останавливаем все процессы MLflow
pkill -f "mlflow ui" 2>/dev/null || true
pkill -f "mlflow server" 2>/dev/null || true

# Удаляем старые данные MLflow
echo "Удаление старых данных MLflow..."
rm -f mlflow.db 2>/dev/null
rm -rf mlruns/ 2>/dev/null

# Создаем чистую структуру
echo "Создание чистой структуры..."
mkdir -p models data/processed data/raw reports mlruns feature_repo/data

# Удаляем старые модели и метаданные
rm -f models/best_run_id.txt models/best_model_info.json 2>/dev/null

echo "✅ MLflow очищен и готов к работе!"
echo ""
echo "Запустите пайплайн заново:"
echo "  bash run_pipeline.sh"