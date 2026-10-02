# Celsius → Fahrenheit ML

A lightweight Flask web app based on the Celsius/Fahrenheit training data in the supplied notebook.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Render deployment

Create a **Web Service** from this GitHub repository.

- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`

No environment variables are required.
