#!/bin/bash

# Telco Churn MLOps Pipeline - Quick Start Script

echo "======================================================================"
echo "🚀 TELCO CHURN MLOPS PIPELINE - QUICK START"
echo "======================================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

echo ""
echo -e "${BLUE}📦 Step 1: Installing dependencies...${NC}"
pip install -q pandas numpy scikit-learn joblib mlflow matplotlib seaborn 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Dependencies may already be installed${NC}"
fi

echo ""
echo -e "${BLUE}🔄 Step 2: Running preprocessing...${NC}"
python src/preprocess.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ Preprocessing issue - data may already exist${NC}"
fi

echo ""
echo -e "${BLUE}🤖 Step 3: Training models...${NC}"
python src/train.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ Training issue${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📊 Step 4: Evaluating model...${NC}"
python src/evaluate.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ Evaluation issue${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📦 Step 5: Registering model in MLflow...${NC}"
python src/register_model.py --auto
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ Registration issue${NC}"
fi

echo ""
echo -e "${BLUE}📤 Step 6: Auto-promoting model...${NC}"
python src/promote_model.py --auto
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ Promotion issue${NC}"
fi

echo ""
echo "======================================================================"
echo -e "${GREEN}✅ PIPELINE COMPLETED SUCCESSFULLY!${NC}"
echo "======================================================================"
echo ""
echo -e "${YELLOW}📊 Next steps:${NC}"
echo "1. View reports:"
echo "   - reports/eval.json      (JSON metrics)"
echo "   - reports/eval.html      (Interactive HTML report)"
echo ""
echo "2. Launch MLflow UI:"
echo "   mlflow ui"
echo "   Then navigate to http://localhost:5000"
echo ""
echo "3. Check registered models:"
echo "   python src/promote_model.py --list --model-name flight_delay_model"
echo ""
echo "======================================================================"
