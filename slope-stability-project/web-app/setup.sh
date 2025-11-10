#!/bin/bash

# Slope Stability Predictor - Setup Script
# Automates the setup process for the web application

set -e  # Exit on error

echo "================================================"
echo "  Slope Stability Predictor - Setup Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/inanotherlife/Mining ANN"
BACKEND_DIR="$BASE_DIR/web-app/backend"
FRONTEND_DIR="$BASE_DIR/web-app/frontend"
MODELS_SOURCE="$BASE_DIR/new/models"

# Check prerequisites
echo -e "${BLUE}[1/5] Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi
NODE_VERSION=$(node --version)
echo -e "${GREEN}✓ Node.js $NODE_VERSION found${NC}"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm not found. Please install npm${NC}"
    exit 1
fi
NPM_VERSION=$(npm --version)
echo -e "${GREEN}✓ npm $NPM_VERSION found${NC}"

echo ""

# Copy model files
echo -e "${BLUE}[2/5] Copying model files...${NC}"

if [ ! -d "$MODELS_SOURCE" ]; then
    echo -e "${RED}❌ Models directory not found: $MODELS_SOURCE${NC}"
    exit 1
fi

mkdir -p "$BACKEND_DIR/models"

if [ -f "$MODELS_SOURCE/gradient_boosting_model.pkl" ]; then
    cp "$MODELS_SOURCE/gradient_boosting_model.pkl" "$BACKEND_DIR/models/"
    echo -e "${GREEN}✓ Copied gradient_boosting_model.pkl${NC}"
else
    echo -e "${RED}❌ gradient_boosting_model.pkl not found${NC}"
    exit 1
fi

if [ -f "$MODELS_SOURCE/xgboost_model.pkl" ]; then
    cp "$MODELS_SOURCE/xgboost_model.pkl" "$BACKEND_DIR/models/"
    echo -e "${GREEN}✓ Copied xgboost_model.pkl${NC}"
else
    echo -e "${RED}❌ xgboost_model.pkl not found${NC}"
    exit 1
fi

if [ -f "$MODELS_SOURCE/scaler.pkl" ]; then
    cp "$MODELS_SOURCE/scaler.pkl" "$BACKEND_DIR/models/"
    echo -e "${GREEN}✓ Copied scaler.pkl${NC}"
else
    echo -e "${RED}❌ scaler.pkl not found${NC}"
    exit 1
fi

echo ""

# Setup backend
echo -e "${BLUE}[3/5] Setting up backend (Flask)...${NC}"

cd "$BACKEND_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate venv and install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo -e "${GREEN}✓ Backend dependencies installed${NC}"
deactivate

echo ""

# Setup frontend
echo -e "${BLUE}[4/5] Setting up frontend (Svelte)...${NC}"

cd "$FRONTEND_DIR"

# Install npm dependencies
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing npm packages (this may take a few minutes)...${NC}"
    npm install --silent
    echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
else
    echo -e "${GREEN}✓ node_modules already exists${NC}"
    echo -e "${YELLOW}Run 'npm install' manually if you want to update${NC}"
fi

echo ""

# Verification
echo -e "${BLUE}[5/5] Verifying installation...${NC}"

# Check backend files
if [ -f "$BACKEND_DIR/app.py" ]; then
    echo -e "${GREEN}✓ Backend app.py found${NC}"
else
    echo -e "${RED}❌ Backend app.py missing${NC}"
fi

# Check model files
if [ -f "$BACKEND_DIR/models/gradient_boosting_model.pkl" ] && \
   [ -f "$BACKEND_DIR/models/xgboost_model.pkl" ] && \
   [ -f "$BACKEND_DIR/models/scaler.pkl" ]; then
    echo -e "${GREEN}✓ All model files present${NC}"
else
    echo -e "${RED}❌ Some model files missing${NC}"
fi

# Check frontend files
if [ -f "$FRONTEND_DIR/package.json" ]; then
    echo -e "${GREEN}✓ Frontend package.json found${NC}"
else
    echo -e "${RED}❌ Frontend package.json missing${NC}"
fi

if [ -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${GREEN}✓ Frontend node_modules installed${NC}"
else
    echo -e "${RED}❌ Frontend node_modules missing${NC}"
fi

echo ""
echo "================================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "================================================"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo -e "1. Start Backend (Terminal 1):"
echo -e "   ${BLUE}cd \"$BACKEND_DIR\"${NC}"
echo -e "   ${BLUE}source venv/bin/activate${NC}"
echo -e "   ${BLUE}python app.py${NC}"
echo ""
echo -e "2. Start Frontend (Terminal 2):"
echo -e "   ${BLUE}cd \"$FRONTEND_DIR\"${NC}"
echo -e "   ${BLUE}npm run dev${NC}"
echo ""
echo -e "3. Open Browser:"
echo -e "   ${BLUE}http://localhost:3000${NC}"
echo ""
echo -e "${GREEN}Happy predicting! 🏔️${NC}"
