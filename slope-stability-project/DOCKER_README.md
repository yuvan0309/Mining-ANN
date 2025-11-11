# 🐳 Docker Deployment - Quick Reference

## One-Command Start

### Linux/Mac:
```bash
./start-docker.sh
```

### Windows:
```cmd
start-docker.bat
```

Or double-click the `start-docker.bat` file in Windows Explorer.

## Manual Start

```bash
docker compose up -d
```

## Access the Application

- **Web Interface**: http://localhost:3000
- **API**: http://localhost:5000
- **API Health**: http://localhost:5000/health

## Common Commands

| Action | Command |
|--------|---------|
| Start | `docker compose up -d` |
| Stop | `docker compose down` |
| Restart | `docker compose restart` |
| View logs | `docker compose logs -f` |
| Check status | `docker compose ps` |
| Rebuild | `docker compose up -d --build` |

## What Gets Installed

The Docker setup automatically installs:

### Backend Container:
- Python 3.13
- Flask 3.0.0
- Flask-CORS
- NumPy, Pandas, Scikit-learn
- XGBoost
- Joblib
- All other dependencies from requirements.txt

### Frontend Container:
- Node.js 20
- Svelte 4.2.0
- Vite 5.4.21
- Axios
- All other dependencies from package.json

## Why Docker?

✅ **Cross-platform**: Works identically on Windows, Linux, and macOS
✅ **Isolated**: No conflicts with existing Python/Node installations
✅ **Reproducible**: Same environment everywhere
✅ **Easy cleanup**: Remove everything with `docker compose down -v`
✅ **No manual setup**: No need to install Python, Node, or manage virtual environments

## Requirements

- Docker Desktop (includes Docker Compose)
- 4GB RAM minimum
- 5GB disk space for images

## Troubleshooting

### Port conflicts?
Edit `docker-compose.yml` and change port mappings:
```yaml
ports:
  - "YOUR_PORT:5000"  # Backend
  - "YOUR_PORT:3000"  # Frontend
```

### Containers not starting?
```bash
docker compose logs -f
```

### Need to reset everything?
```bash
docker compose down -v
docker system prune -a
./start-docker.sh  # Start fresh
```

## Full Documentation

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for complete documentation.
