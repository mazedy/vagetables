# Deploying Image Classification API to Render

This guide will help you deploy your AI-powered image classification API to Render.

## Prerequisites

- A [Render account](https://render.com) (free tier available)
- Git repository with your code (GitHub, GitLab, or Bitbucket)

## Deployment Steps

### Option 1: Deploy with render.yaml (Recommended)

1. **Push your code to a Git repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Image Classification API"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Connect to Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click **"New +"** → **"Blueprint"**
   - Connect your Git repository
   - Render will automatically detect `render.yaml` and configure the service

3. **Wait for deployment**
   - First deployment takes 5-10 minutes (downloading model ~330MB)
   - Monitor logs for "Model loaded successfully!"

### Option 2: Manual Web Service Setup

1. **Create a new Web Service**
   - Go to Render Dashboard
   - Click **"New +"** → **"Web Service"**
   - Connect your Git repository

2. **Configure the service**
   - **Name**: `image-classification-api`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free or Starter (Starter recommended for better performance)

3. **Add Environment Variables** (Optional but recommended)
   - `TRANSFORMERS_CACHE`: `/opt/render/project/.cache`
   - `HF_HOME`: `/opt/render/project/.cache`

4. **Deploy**
   - Click **"Create Web Service"**
   - Wait for deployment to complete

## Important Notes

### Memory Requirements
- **Free tier**: 512MB RAM (may struggle with model loading)
- **Starter tier**: 2GB RAM (recommended for stable performance)
- The Vision Transformer model requires ~1GB RAM when loaded

### First Deployment
- Initial deployment downloads the model (~330MB)
- Subsequent deployments use cached model
- Model is cached in `/opt/render/project/.cache`

### Performance Considerations
- Free tier services spin down after 15 minutes of inactivity
- First request after spin-down will be slow (model reload)
- Starter tier keeps service always running

## Testing Your Deployment

Once deployed, Render will provide a URL like: `https://image-classification-api-xxxx.onrender.com`

### Test the API

1. **Health Check**
   ```bash
   curl https://your-app-url.onrender.com/health
   ```

2. **API Documentation**
   Visit: `https://your-app-url.onrender.com/docs`

3. **Web Interface**
   Visit: `https://your-app-url.onrender.com/`

4. **Test Classification**
   ```bash
   curl -X POST "https://your-app-url.onrender.com/classify" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"
   ```

## Monitoring

- **Logs**: Available in Render Dashboard → Your Service → Logs
- **Metrics**: Dashboard shows CPU, Memory, and Request metrics
- **Health Check**: Configured at `/health` endpoint

## Troubleshooting

### Out of Memory Errors
- Upgrade to Starter plan (2GB RAM)
- Model requires significant memory on first load

### Slow First Request
- Normal behavior on free tier (service spins down)
- Consider Starter plan for always-on service

### Model Download Fails
- Check logs for network errors
- Verify Hugging Face is accessible
- May need to retry deployment

## Updating Your Deployment

Render automatically redeploys when you push to your connected Git branch:

```bash
git add .
git commit -m "Update API"
git push
```

## Cost Estimate

- **Free Tier**: $0/month (512MB RAM, spins down after inactivity)
- **Starter Tier**: $7/month (2GB RAM, always on, recommended)

## Support

For issues or questions:
- Render Documentation: https://render.com/docs
- Render Community: https://community.render.com
