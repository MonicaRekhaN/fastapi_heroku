web: gunicorn main:app  --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker --timeout 1200 --preload
heroku ps:scale web=1
