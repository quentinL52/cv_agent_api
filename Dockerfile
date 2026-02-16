FROM python:3.11-slim

# Force l'affichage des logs en temps réel
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Création de l'utilisateur
RUN useradd -m -u 1000 user
USER user

# --- CORRECTION ICI ---
# 1. On garde le PATH pour les exécutables
ENV PATH="/home/user/.local/bin:$PATH"
# 2. On ajoute le PYTHONPATH pour que Python trouve les bibliothèques installées (C'est la ligne manquante !)
ENV PYTHONPATH="/home/user/.local/lib/python3.11/site-packages"

WORKDIR /app

# Installation des dépendances
COPY --chown=user ./requirements.txt requirements.txt
# Ajout explicite de --user pour être sûr de l'endroit d'installation
RUN pip install --user --no-cache-dir --upgrade -r requirements.txt

# Copie du code
COPY --chown=user . /app

EXPOSE 7860

# Lancement
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]