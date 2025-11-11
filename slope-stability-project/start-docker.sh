#!/bin/bash

# Slope Stability Prediction - Docker Startup Script (Linux/Mac)

echo "=========================================="
echo "Slope Stability Prediction - Docker Setup"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not available!"
    echo "Please update Docker Desktop to the latest version"
    exit 1
fi

echo "✅ Docker is installed"
echo ""

# Check if models exist
if [ ! -f "web-app/backend/models/best_model_gradient_boosting.pkl" ]; then
    echo "⚠️  Warning: Model files not found in web-app/backend/models/"
    echo "Please ensure the following files exist:"
    echo "  - best_model_gradient_boosting.pkl"
    echo "  - best_model_xgboost.pkl"
    echo "  - scaler.pkl"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🚀 Starting Slope Stability Prediction application..."
echo ""

# Stop any existing containers
echo "📦 Stopping existing containers (if any)..."
docker compose down 2>/dev/null

# Build and start containers
echo "🏗️  Building Docker images (this may take a few minutes on first run)..."
docker compose up -d --build

# Wait for containers to be healthy
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check status
echo ""
echo "📊 Container Status:"
docker compose ps

# Show logs
echo ""
echo "📝 Recent Logs:"
docker compose logs --tail=20

echo ""
echo "=========================================="
echo "✅ Application is running!"
echo "=========================================="
echo ""
echo "🌐 Access the application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:5000"
echo ""
echo "📊 Useful commands:"
echo "   View logs: docker compose logs -f"
echo "   Stop app: docker compose down"
echo "   Restart: docker compose restart"
echo ""
echo "📖 For more info, see DOCKER_GUIDE.md"
echo "=========================================="
