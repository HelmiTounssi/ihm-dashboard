# Utiliser une image Docker basée sur Python
FROM python:3.8

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
# Installer les dépendances de l'application Streamlit
RUN pip install --no-cache-dir streamlit
RUN pip install --ignore-installed -r requirements.txt
RUN pip install --upgrade pandas

# Exposer le port 5000
EXPOSE 8501

# Commande pour exécuter l'application Streamlit
CMD ["streamlit", "run", "dashboard/main.py"]