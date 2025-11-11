# Slope Stability Prediction - Docker Deployment Guide

This guide will help you run the Slope Stability Prediction application using Docker on any platform (Windows, Linux, macOS).

## 📋 Prerequisites

1. **Install Docker Desktop**:
   - **Windows**: Download from [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
   - **Linux**: Follow instructions for your distribution at [Docker for Linux](https://docs.docker.com/engine/install/)
   - **macOS**: Download from [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)

2. **Verify Docker Installation**:
   ```bash
   docker --version
   docker compose version
   ```

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Navigate to project directory**:
   ```bash
   cd "/home/inanotherlife/Mining ANN/slope-stability-project"
   ```

2. **Start both backend and frontend**:
   ```bash
   docker compose up -d
   ```

3. **Check status**:
   ```bash
   docker compose ps
   ```

4. **View logs**:
   ```bash
   # All services
   docker compose logs -f
   
   # Backend only
   docker compose logs -f backend
   
   # Frontend only
   docker compose logs -f frontend
   ```

5. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

6. **Stop the application**:
   ```bash
   docker compose down
   ```

### Option 2: Manual Docker Build

#### Backend Only:
```bash
# Build
docker build -f Dockerfile.backend -t slope-backend .

# Run
docker run -d -p 5000:5000 \
  -v $(pwd)/web-app/backend/models:/app/models:ro \
  --name slope-backend \
  slope-backend

# Check logs
docker logs -f slope-backend
```

#### Frontend Only:
```bash
# Build
docker build -f Dockerfile.frontend -t slope-frontend .

# Run
docker run -d -p 3000:3000 \
  -e VITE_API_URL=http://localhost:5000 \
  --name slope-frontend \
  slope-frontend

# Check logs
docker logs -f slope-frontend
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Backend
FLASK_ENV=production
FLASK_PORT=5000

# Frontend
VITE_API_URL=http://localhost:5000
VITE_PORT=3000
```

### Custom Ports

Edit `docker-compose.yml` to change ports:

```yaml
services:
  backend:
    ports:
      - "YOUR_PORT:5000"  # Change YOUR_PORT
  frontend:
    ports:
      - "YOUR_PORT:3000"  # Change YOUR_PORT
```

## 🛠️ Useful Commands

### View Running Containers
```bash
docker compose ps
```

### Restart Services
```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart backend
docker compose restart frontend
```

### Rebuild After Code Changes
```bash
# Rebuild and restart
docker compose up -d --build

# Rebuild specific service
docker compose up -d --build backend
```

### Stop and Remove Everything
```bash
docker compose down -v
```

### Access Container Shell
```bash
# Backend
docker compose exec backend /bin/bash

# Frontend
docker compose exec frontend /bin/sh
```

## 📊 Health Checks

Both services have built-in health checks:

```bash
# Check backend health
curl http://localhost:5000/health

# Check frontend (should return HTML)
curl http://localhost:3000
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port (Linux/Mac)
sudo lsof -i :5000
sudo lsof -i :3000

# Find process using port (Windows - PowerShell)
netstat -ano | findstr :5000
netstat -ano | findstr :3000

# Change ports in docker-compose.yml
```

### Models Not Found
Ensure model files exist:
```bash
ls -la web-app/backend/models/
# Should contain:
# - best_model_gradient_boosting.pkl
# - best_model_xgboost.pkl
# - scaler.pkl
```

### Container Keeps Restarting
```bash
# Check logs
docker compose logs backend
docker compose logs frontend

# Check container status
docker compose ps
```

### Permission Issues (Linux)
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Logout and login again
```

## 🔄 Updates and Maintenance

### Update Application
```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose up -d --build
```

### Clean Up Docker System
```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything unused
docker system prune -a --volumes
```

## 📦 Production Deployment

For production deployment, consider:

1. **Use production-ready web servers**:
   - Backend: Gunicorn instead of Flask dev server
   - Frontend: Nginx to serve static files

2. **Enable HTTPS**:
   - Add SSL certificates
   - Use reverse proxy (Nginx/Traefik)

3. **Add monitoring**:
   - Container health monitoring
   - Application performance monitoring

4. **Use Docker secrets** for sensitive data

5. **Set up logging**:
   - Centralized logging (ELK stack, etc.)
   - Log rotation

## 🌐 Platform-Specific Notes

### Windows
- Use PowerShell or Command Prompt
- Docker Desktop must be running
- WSL2 backend recommended for better performance
- Use forward slashes in paths: `//c/Users/...`

### Linux
- May need `sudo` for Docker commands
- Add user to docker group to avoid sudo
- Check firewall rules if ports are blocked

### macOS
- Docker Desktop must be running
- M1/M2 Macs: Docker handles ARM architecture automatically
- File sharing permissions in Docker Desktop settings

## 📞 Support

For issues or questions:
1. Check logs: `docker compose logs -f`
2. Verify containers: `docker compose ps`
3. Check health: `curl http://localhost:5000/health`
4. Review this documentation

## 📝 Notes

- First build may take 5-10 minutes
- Subsequent starts are much faster
- Models are loaded from `web-app/backend/models/`
- Data is not persisted between container rebuilds
- Both services auto-restart on failure
