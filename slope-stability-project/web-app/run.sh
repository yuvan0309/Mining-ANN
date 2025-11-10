#!/bin/bash

# Slope Stability Predictor - Run Script
# Opens two terminal windows to run backend and frontend

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

BASE_DIR="/home/inanotherlife/Mining ANN"
BACKEND_DIR="$BASE_DIR/web-app/backend"
FRONTEND_DIR="$BASE_DIR/web-app/frontend"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Slope Stability Predictor - Launch Script${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if models exist
if [ ! -f "$BACKEND_DIR/models/gradient_boosting_model.pkl" ]; then
    echo -e "${YELLOW}⚠️  Model files not found. Run setup.sh first:${NC}"
    echo -e "   ${BLUE}./setup.sh${NC}"
    exit 1
fi

# Check if backend venv exists
if [ ! -d "$BACKEND_DIR/venv" ]; then
    echo -e "${YELLOW}⚠️  Backend not set up. Run setup.sh first:${NC}"
    echo -e "   ${BLUE}./setup.sh${NC}"
    exit 1
fi

# Check if frontend node_modules exists
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}⚠️  Frontend not set up. Run setup.sh first:${NC}"
    echo -e "   ${BLUE}./setup.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"
echo ""

# Function to detect terminal emulator
detect_terminal() {
    if command -v gnome-terminal &> /dev/null; then
        echo "gnome-terminal"
    elif command -v konsole &> /dev/null; then
        echo "konsole"
    elif command -v xfce4-terminal &> /dev/null; then
        echo "xfce4-terminal"
    elif command -v xterm &> /dev/null; then
        echo "xterm"
    else
        echo "none"
    fi
}

TERMINAL=$(detect_terminal)

if [ "$TERMINAL" = "none" ]; then
    echo -e "${YELLOW}⚠️  Could not detect terminal emulator${NC}"
    echo ""
    echo -e "${BLUE}Please run these commands in separate terminals:${NC}"
    echo ""
    echo -e "${YELLOW}Terminal 1 (Backend):${NC}"
    echo -e "  cd \"$BACKEND_DIR\""
    echo -e "  source venv/bin/activate"
    echo -e "  python app.py"
    echo ""
    echo -e "${YELLOW}Terminal 2 (Frontend):${NC}"
    echo -e "  cd \"$FRONTEND_DIR\""
    echo -e "  npm run dev"
    echo ""
    exit 0
fi

echo -e "${BLUE}Launching backend in new terminal...${NC}"

# Launch backend in new terminal
case $TERMINAL in
    "gnome-terminal")
        gnome-terminal -- bash -c "cd '$BACKEND_DIR' && source venv/bin/activate && echo -e '\033[0;32m🚀 Starting Backend (Flask API)...\033[0m' && python app.py; exec bash"
        ;;
    "konsole")
        konsole -e bash -c "cd '$BACKEND_DIR' && source venv/bin/activate && echo -e '\033[0;32m🚀 Starting Backend (Flask API)...\033[0m' && python app.py; exec bash" &
        ;;
    "xfce4-terminal")
        xfce4-terminal -e "bash -c 'cd \"$BACKEND_DIR\" && source venv/bin/activate && echo -e \"\033[0;32m🚀 Starting Backend (Flask API)...\033[0m\" && python app.py; exec bash'" &
        ;;
    "xterm")
        xterm -e "bash -c 'cd \"$BACKEND_DIR\" && source venv/bin/activate && echo -e \"\033[0;32m🚀 Starting Backend (Flask API)...\033[0m\" && python app.py; exec bash'" &
        ;;
esac

sleep 2

echo -e "${BLUE}Launching frontend in new terminal...${NC}"

# Launch frontend in new terminal
case $TERMINAL in
    "gnome-terminal")
        gnome-terminal -- bash -c "cd '$FRONTEND_DIR' && echo -e '\033[0;32m🚀 Starting Frontend (Svelte)...\033[0m' && npm run dev; exec bash"
        ;;
    "konsole")
        konsole -e bash -c "cd '$FRONTEND_DIR' && echo -e '\033[0;32m🚀 Starting Frontend (Svelte)...\033[0m' && npm run dev; exec bash" &
        ;;
    "xfce4-terminal")
        xfce4-terminal -e "bash -c 'cd \"$FRONTEND_DIR\" && echo -e \"\033[0;32m🚀 Starting Frontend (Svelte)...\033[0m\" && npm run dev; exec bash'" &
        ;;
    "xterm")
        xterm -e "bash -c 'cd \"$FRONTEND_DIR\" && echo -e \"\033[0;32m🚀 Starting Frontend (Svelte)...\033[0m\" && npm run dev; exec bash'" &
        ;;
esac

sleep 3

echo ""
echo -e "${GREEN}✅ Both servers starting...${NC}"
echo ""
echo -e "${YELLOW}Wait for servers to start, then open:${NC}"
echo -e "   ${BLUE}http://localhost:3000${NC}"
echo ""
echo -e "${YELLOW}Backend API:${NC}"
echo -e "   ${BLUE}http://localhost:5000${NC}"
echo ""
echo -e "${GREEN}Press Ctrl+C in each terminal window to stop servers${NC}"
