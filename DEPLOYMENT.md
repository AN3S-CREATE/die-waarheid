# Die Waarheid Deployment Guide

## Overview

This guide covers the deployment of Die Waarheid forensic analysis platform in various environments.

## Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB+ free space
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.10-3.12
- **GPU**: Optional, for enhanced AI processing

### API Keys Required
- Google Gemini API key
- Hugging Face token (optional)

## Deployment Options

### 1. Local Development

```bash
# Clone repository
git clone https://github.com/your-org/die-waarheid.git
cd die-waarheid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp die_waarheid/.env.example die_waarheid/.env
# Edit .env with your API keys

# Run the application
streamlit run die_waarheid/app.py
```

### 2. Docker Deployment

#### Single Container

```bash
# Build image
docker build -t die-waarheid .

# Run container
docker run -d \
  --name die-waarheid \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  die-waarheid
```

#### Docker Compose

```bash
# Basic deployment
docker-compose up -d

# Production deployment with database
docker-compose --profile production up -d

# View logs
docker-compose logs -f die-waarheid

# Stop services
docker-compose down
```

### 3. Cloud Deployment

#### AWS ECS

1. Create ECR repository
2. Push Docker image
3. Create ECS task definition
4. Deploy service

```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com
docker build -t die-waarheid .
docker tag die-waarheid:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/die-waarheid:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/die-waarheid:latest
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/die-waarheid
gcloud run deploy die-waarheid --image gcr.io/PROJECT-ID/die-waarheid --platform managed
```

#### Azure Container Instances

```bash
# Create resource group
az group create --name die-waarheid-rg --location eastus

# Deploy container
az container create \
  --resource-group die-waarheid-rg \
  --name die-waarheid \
  --image die-waarheid/die-waarheid:latest \
  --ports 8501 \
  --cpu 2 \
  --memory 8
```

## Environment Configuration

### Environment Variables

```bash
# Application
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# AI Services
GEMINI_API_KEY=your_gemini_api_key
HUGGINGFACE_TOKEN=your_huggingface_token

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/die_waarheid

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Production Settings

1. **Security**
   - Use HTTPS in production
   - Set strong passwords
   - Enable firewall rules
   - Use secrets management

2. **Performance**
   - Enable caching
   - Use load balancer
   - Configure auto-scaling
   - Monitor resources

3. **Backup**
   - Regular database backups
   - File storage backup
   - Configuration backup

## Monitoring and Logging

### Health Checks

```bash
# Application health
curl http://localhost:8501/_stcore/health

# System metrics
curl http://localhost:8501/api/metrics
```

### Log Management

```bash
# View logs
docker-compose logs -f die-waarheid

# Log files location
tail -f data/logs/die_waarheid_YYYYMMDD.log
```

### Monitoring Stack

Optional: Deploy with monitoring stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
  
  loki:
    image: grafana/loki
    ports:
      - "3100:3100"
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find process
   lsof -i :8501
   # Kill process
   kill -9 <PID>
   ```

2. **Memory issues**
   ```bash
   # Increase swap
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Permission errors**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER data/
   chmod -R 755 data/
   ```

4. **API quota exceeded**
   - Check API key status
   - Upgrade plan if needed
   - Implement caching

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug flags
streamlit run die_waarheid/app.py --logger.level=debug
```

## Security Considerations

1. **API Keys**
   - Never commit to version control
   - Use environment variables
   - Rotate regularly

2. **File Uploads**
   - Validate file types
   - Scan for malware
   - Limit file sizes

3. **Network Security**
   - Use VPN for remote access
   - Configure firewall
   - Enable DDoS protection

4. **Data Privacy**
   - Encrypt sensitive data
   - Follow GDPR/CCPA
   - Regular security audits

## Maintenance

### Regular Tasks

1. **Daily**
   - Check system health
   - Review error logs
   - Monitor resource usage

2. **Weekly**
   - Update dependencies
   - Clean temp files
   - Backup data

3. **Monthly**
   - Security updates
   - Performance review
   - Capacity planning

### Updates

```bash
# Update application
git pull origin main
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Support

For deployment issues:
1. Check logs first
2. Review troubleshooting guide
3. Search GitHub issues
4. Contact support team

## Appendix

### Docker Commands Reference

```bash
# Build image
docker build -t die-waarheid:latest .

# Run container
docker run -d --name die-waarheid -p 8501:8501 die-waarheid

# Stop container
docker stop die-waarheid

# Remove container
docker rm die-waarheid

# View logs
docker logs -f die-waarheid

# Execute in container
docker exec -it die-waarheid bash
```

### Kubernetes Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: die-waarheid
spec:
  replicas: 3
  selector:
    matchLabels:
      app: die-waarheid
  template:
    metadata:
      labels:
        app: die-waarheid
    spec:
      containers:
      - name: die-waarheid
        image: die-waarheid/die-waarheid:latest
        ports:
        - containerPort: 8501
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: die-waarheid-secrets
              key: gemini-api-key
---
apiVersion: v1
kind: Service
metadata:
  name: die-waarheid-service
spec:
  selector:
    app: die-waarheid
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```
