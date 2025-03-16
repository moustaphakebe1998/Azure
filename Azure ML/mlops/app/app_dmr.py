# App classification de commentaires

import streamlit as st
import urllib.request
import json
import os
import pandas as pd
import logging
from typing import Dict, Any, List, Union
import time
from datetime import datetime
import fasttext
import plotly.express as px
import warnings
import numpy as np

# Configuration
API_URL = 'http://.../api/v1/service/dmr-prediction/score'
LOG_DIR = '/app/logs'
LOG_FILE = f'{LOG_DIR}/app.log'
FASTTEXT_MODEL_PATH = '/app/modele_langdetect/lid.176.bin'

# Légende des classes avec emojis associés
CLASSES = {
    0: {'name': 'Activités commerciales et professionnelles', 'emoji': '🏪'},
    1: {'name': 'Arbres, végétaux et animaux', 'emoji': '🌳'},
    2: {'name': 'Autos, motos, vélos, trottinettes...', 'emoji': '🚗'},
    3: {'name': 'Eau', 'emoji': '💧'},
    4: {'name': 'Graffitis, tags, affiches et autocollants', 'emoji': '🖌️'},
    5: {'name': 'Mobiliers urbains', 'emoji': '🪑'},
    6: {'name': 'Objets abandonnés', 'emoji': '📦'},
    7: {'name': 'Propreté', 'emoji': '🧹'},
    8: {'name': 'Voirie et espace public', 'emoji': '🛣️'},
    9: {'name': 'Éclairage / Électricité', 'emoji': '💡'}
}

# Exemples de commentaires prédéfinis
EXAMPLE_COMMENTS = [
    "Des graffitis sur le mur de l'école",
    "Arbre tombé sur la route après l'orage",
    "Lampadaire qui ne s'allume plus depuis 3 jours",
    "Poubelle renversée avec déchets éparpillés"
]

# Load FastText model
@st.cache_resource
def load_fasttext_model():
    """
    Charge le modèle FastText pour la détection de langue.
    
    Returns:
        Le modèle FastText chargé
    """
    try:
        # Disable verbose output from FastText
        fasttext.FastText.eprint = lambda x: None
        model = fasttext.load_model(FASTTEXT_MODEL_PATH)
        logging.info("Modèle FastText chargé avec succès.")
        return model
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle FastText: {str(e)}")
        st.error(f"Erreur lors du chargement du modèle de détection de langue: {str(e)}")
        return None

def detect_language(text: str) -> str:
    """
    Détecte la langue du texte fourni en utilisant FastText.
    
    Args:
        text: Le texte dont on veut détecter la langue
        
    Returns:
        Le code de la langue détectée (fr, en, es, etc.) ou "unknown" si une erreur se produit.
    """
    try:
        # Vérifier que le texte n'est pas vide
        if not text or text.isspace():
            logging.warning("Texte vide pour la détection de langue")
            return "unknown"
        
        # Obtenir le modèle FastText
        model = load_fasttext_model()
        if model is None:
            logging.warning("Le modèle de détection de langue n'a pas été chargé.")
            return "modèle non chargé"
        
        # Préparer le texte pour FastText
        clean_text = text.replace('\n', ' ').strip()
        
        try:
            # Solution pour contourner le problème de NumPy 2.0
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
                warnings.filterwarnings("ignore", message="Unable to avoid copy while creating an array")
                
                # Prédiction avec fasttext
                predictions = model.predict(clean_text)
        except Exception as e:
            logging.warning(f"Erreur lors de la prédiction FastText: {str(e)}")
            # Fallback en cas d'erreur
            try:
                # Méthode alternative
                lang_label = model.predict(clean_text, k=1)[0][0]
                logging.info(f"Utilisation de la méthode alternative pour la détection de langue: {lang_label}")
                return lang_label.replace('__label__', '')
            except Exception as e2:
                logging.error(f"Échec de la méthode alternative: {str(e2)}")
                return "unknown"
        
        # Extraire le code de langue du résultat
        lang_code = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]  # Confiance de la prédiction
        
        logging.info(f"Langue détectée: {lang_code} (confiance: {confidence:.4f})")
        
        return lang_code
    except Exception as e:
        logging.error(f"Erreur non gérée lors de la détection de langue: {str(e)}")
        return "unknown"


# Fonction pour initialiser la configuration de la page
def setup_page():
    """Configure l'apparence de la page Streamlit."""
    st.set_page_config(
        page_title="Classificateur de Commentaires",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        margin: 10px 0px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .example-button button {
        background-color: #2196F3;
        font-size: 0.85em;
        padding: 8px 12px;
        margin: 5px 0px;
    }
    .example-button button:hover {
        background-color: #0b7dda;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f1f1f1;
        margin: 20px 0px;
        border-left: 5px solid #4CAF50;
    }
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd;
        margin: 20px 0px;
        border-left: 5px solid #ffc107;
    }
    h1 {
        color: #2C3E50;
        margin-bottom: 30px;
    }
    h2, h3 {
        color: #34495E;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
    .class-badge {
        display: inline-block;
        padding: 5px 10px;
        background-color: #E5F3FF;
        border-radius: 15px;
        font-size: 0.9em;
        margin: 3px;
    }
    .last-update {
        font-size: 0.8em;
        color: #7F8C8D;
        margin-top: 30px;
    }
    .probability-bar {
        height: 25px;
        background: linear-gradient(90deg, #4CAF50 var(--width), #f1f1f1 var(--width));
        border-radius: 5px;
        margin-top: 5px;
        position: relative;
    }
    .probability-value {
        position: absolute;
        right: 10px;
        top: 3px;
        font-weight: bold;
        color: #333;
    }
    .language-info {
        font-size: 0.9em;
        color: #666;
        font-style: italic;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def setup_logging() -> None:
    """Configure le système de journalisation."""
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def call_prediction_api(commentaire: str) -> dict:
    """
    Appelle l'API de prédiction avec le commentaire fourni.
    
    Args:
        commentaire: Le texte à classifier
        
    Returns:
        Le résultat JSON de l'API
        
    Raises:
        Exception: En cas d'erreur de connexion ou de réponse de l'API
    """
    data = {
        "Inputs": {
            "data": [
                {
                    "commentaire_usager": commentaire,
                }
            ]
        },
        "GlobalParameters": {
            "method": "predict_proba"
        }
    }
    
    body = str.encode(json.dumps(data))
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(API_URL, body, headers)
    
    logging.info(f"Envoi du commentaire: {commentaire}")
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        result_json = json.loads(result)
        logging.info(f"Réponse reçue: {result_json}")
        return result_json
    except urllib.error.HTTPError as error:
        error_message = error.read().decode("utf8", 'ignore')
        logging.error(f"Erreur HTTP {error.code}: {error_message}")
        raise Exception(f"La requête a échoué avec le code: {error.code}\n{error_message}")
    except Exception as e:
        logging.error(f"Erreur inattendue: {str(e)}")
        raise

def parse_prediction_result(result_json: dict) -> Dict[int, float]:
    """
    Analyse le résultat de l'API pour extraire les probabilités de chaque classe.
    
    Args:
        result_json: La réponse JSON de l'API
        
    Returns:
        Un dictionnaire avec les indices de classe comme clés et les probabilités comme valeurs
        
    Raises:
        ValueError: Si le format de la réponse n'est pas reconnu
    """
    if "Results" not in result_json:
        raise ValueError("Format de réponse incorrect: 'Results' manquant")
    
    results = result_json["Results"]
    
    # Gérer le cas d'un tableau imbriqué [[0.1, 0.2, ...]]
    if isinstance(results, list) and len(results) == 1 and isinstance(results[0], list):
        probs = results[0]
        return {i: float(prob) for i, prob in enumerate(probs)}
    
    # Si l'API renvoie un tableau de probabilités [0.1, 0.2, ...]
    elif isinstance(results, list) and all(isinstance(x, (int, float)) for x in results):
        return {i: float(prob) for i, prob in enumerate(results)}
    
    # Si l'API renvoie un dictionnaire avec des indices comme clés
    elif isinstance(results, dict):
        probabilities = {}
        for idx_str, prob in results.items():
            try:
                idx = int(idx_str)
                probabilities[idx] = float(prob)
            except (ValueError, TypeError):
                logging.warning(f"Valeur ou clé non attendue dans les résultats: {idx_str}: {prob}")
        
        if probabilities:
            return probabilities
    
    # Si le format ne correspond à aucun des cas ci-dessus
    logging.error(f"Format de réponse non reconnu: {results}")
    raise ValueError(f"Format de réponse non reconnu pour predict_proba: {results}")

def add_result_to_history(commentaire: str, probabilities: Dict[int, float], lang_code: str) -> None:
    """
    Ajoute un résultat à l'historique des prédictions avec les probabilités.
    
    Args:
        commentaire: Le commentaire analysé
        probabilities: Dictionnaire des probabilités par classe
        lang_code: Code de la langue détectée
    """
    # Trouver la classe avec la probabilité la plus élevée
    class_index = max(probabilities, key=probabilities.get)
    class_info = CLASSES.get(class_index, {"name": "Classe inconnue", "emoji": "❓"})
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Créer un dictionnaire avec les informations de base
    result_dict = {
        'Commentaire': commentaire, 
        'Classe prédite': f"{class_info['emoji']} {class_info['name']}", 
        'Probabilité': f"{probabilities[class_index]:.2f}",
        'Langue': lang_code,
        'Date': timestamp
    }
    
    # Ajouter les probabilités des 3 premières classes
    top_classes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    for idx, (class_idx, prob) in enumerate(top_classes):
        class_info = CLASSES.get(class_idx, {"name": "Classe inconnue", "emoji": "❓"})
        result_dict[f"Top {idx+1}"] = f"{class_info['emoji']} {class_info['name']} ({prob:.2f})"
    
    st.session_state['results_list'].append(result_dict)
    logging.info(f"Résultat ajouté à l'historique: {commentaire} -> {class_info['name']} (prob: {probabilities[class_index]:.2f}, langue: {lang_code})")

def display_results() -> None:
    """Affiche le tableau des résultats et le bouton de téléchargement."""
    if st.session_state['results_list']:
        st.markdown("### 📊 Historique des prédictions")
        
        # Création du DataFrame pour l'affichage
        df = pd.DataFrame(st.session_state['results_list'])
        
        # Tri par date la plus récente
        if 'Date' in df.columns:
            df = df.sort_values(by='Date', ascending=False)
        
        # Affichage du tableau
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            # Convertir le DataFrame en CSV
            csv = df.to_csv(index=False, sep=";")
            
            # Bouton de téléchargement
            st.download_button(
                label="📥 Télécharger le CSV",
                data=csv,
                file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        with col2:
            if len(df) > 0:
                st.markdown(f"<p class='last-update'>Dernière mise à jour: {df['Date'].iloc[0]}</p>", unsafe_allow_html=True)

def show_sidebar() -> None:
    """Affiche la barre latérale avec les informations utiles."""
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/000000/document.png", width=100)
        st.markdown("## 🔍 Classification automatique")
        st.markdown("Cet outil vous permet de classifier automatiquement des commentaires en catégories prédéfinies.")
        
        st.markdown("### 📋 Classes disponibles")
        
        for index, class_info in CLASSES.items():
            st.markdown(
                f"<div class='class-badge'>{class_info['emoji']} {index}: {class_info['name']}</div>", 
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        st.markdown("### ℹ️ Note importante")
        st.info("Cette application n'accepte actuellement que les commentaires en français.")
        
        # Informations sur FastText
        st.markdown("### 🌐 Détection de langue")
        st.markdown("Cette application utilise FastText pour détecter automatiquement la langue des commentaires.")
        
        if 'last_response' in st.session_state:
            show_advanced = st.checkbox("⚙️ Afficher les données techniques")
            if show_advanced:
                st.markdown("### Réponse API brute")
                st.json(st.session_state['last_response'])
        
        st.markdown("---")
        st.markdown("### 🎚️ Paramètres d'affichage")
        
        # Paramètre pour le nombre de classes à afficher dans le graphique
        if 'top_n_classes' not in st.session_state:
            st.session_state['top_n_classes'] = 5
        
        st.session_state['top_n_classes'] = st.slider(
            "Nombre de classes à afficher", 
            min_value=1, 
            max_value=len(CLASSES), 
            value=st.session_state['top_n_classes'],
            step=1
        )

def show_example_buttons():
    """
    Affiche des boutons d'exemples pour faciliter les tests.
    Les boutons mettent à jour directement la zone de texte.
    """
    st.markdown("### 💡 Exemples de commentaires")
    
    columns = st.columns(len(EXAMPLE_COMMENTS))
    
    for i, (col, example) in enumerate(zip(columns, EXAMPLE_COMMENTS)):
        with col:
            st.markdown(f'<div class="example-button">', unsafe_allow_html=True)
            if st.button(f"Exemple {i+1}", key=f"example_{i}"):
                st.session_state['comment_input'] = example
            st.markdown('</div>', unsafe_allow_html=True)

def show_probability_chart(probabilities: Dict[int, float]):
    """
    Affiche un graphique des probabilités pour chaque classe.
    
    Args:
        probabilities: Dictionnaire des probabilités par classe
    """
    # Créer un DataFrame pour le graphique
    top_n = st.session_state['top_n_classes']
    top_classes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    chart_data = []
    for class_idx, prob in top_classes:
        class_info = CLASSES.get(class_idx, {"name": "Classe inconnue", "emoji": "❓"})
        chart_data.append({
            "Classe": f"{class_info['emoji']} {class_info['name']}",
            "Probabilité": prob
        })
    
    df_chart = pd.DataFrame(chart_data)
    
    # Créer un graphique à barres horizontal avec Plotly
    fig = px.bar(
        df_chart, 
        x="Probabilité", 
        y="Classe",
        orientation='h',
        title="Probabilités par classe",
        color="Probabilité",
        color_continuous_scale="Viridis",
        range_color=[0, 1],
        height=50 + (50 * len(chart_data))
    )
    
    fig.update_layout(
        xaxis_title="Probabilité",
        yaxis_title="",
        xaxis=dict(range=[0, 1]),
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main() -> None:
    """Fonction principale de l'application."""
    setup_page()
    
    # Titre et description
    st.markdown("# 📝 Classificateur de Commentaires")
    st.markdown("Outil d'analyse et de classification automatique des signalements et commentaires")
    
    # Initialisation de l'état de session
    if 'results_list' not in st.session_state:
        st.session_state['results_list'] = []
    
    if 'comment_input' not in st.session_state:
        st.session_state['comment_input'] = "Des Autocollants sur mur"
    
    # Précharger le modèle FastText
    load_fasttext_model()
    
    # Affichage de la barre latérale
    show_sidebar()
    
    # Section des boutons d'exemples - affichés avant le champ de texte
    # pour que l'exemple choisi s'affiche immédiatement dans le champ
    show_example_buttons()
    
    # Formulaire de saisie
    with st.container():
        st.markdown("### 📝 Saisir un commentaire à classifier")
        
        # Champ de saisie de texte
        # La valeur est maintenant liée à la variable de session mise à jour par les boutons d'exemples
        commentaire = st.text_area(
            "Commentaire", 
            value=st.session_state['comment_input'],
            height=100,
            key="comment_text"
        )
        
        # Mettre à jour la session à chaque changement du champ texte
        st.session_state['comment_input'] = commentaire
    
    # Traitement de la prédiction
    if st.button("🔍 Analyser le commentaire"):
        if not commentaire.strip():
            st.warning("⚠️ Veuillez saisir un commentaire à analyser.")
        else:
            with st.spinner("🔄 Analyse en cours..."):
                try:
                    # Vérification de la langue avec FastText
                    lang_code = detect_language(commentaire)
                    logging.info(f"Langue détectée avec FastText: {lang_code}")
                    
                    # Affichage de la langue détectée
                    st.markdown(f"<p class='language-info'>Langue détectée: {lang_code.upper()}</p>", unsafe_allow_html=True)
                    
                    if lang_code != "fr":
                        # Affichage d'un message d'erreur si la langue n'est pas le français
                        st.markdown(
                            f"""
                            <div class="warning-box">
                                <h3>⚠️ Langue non supportée</h3>
                                <p>Cette application n'accepte actuellement que les commentaires en français.</p>
                                <p>Veuillez traduire votre commentaire en français ou réessayer ultérieurement.</p>
                                <p><em>Langue détectée: {lang_code.upper()}</em></p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        # Effet de chargement pour une meilleure expérience utilisateur
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Appel API
                        result_json = call_prediction_api(commentaire)
                        st.session_state['last_response'] = result_json
                        
                        # Affichage du résultat brut pour debug
                        if st.checkbox("Voir la réponse brute de l'API"):
                            st.json(result_json)
                        
                        # Extraction et traitement des probabilités
                        probabilities = parse_prediction_result(result_json)
                        
                        # Trouver la classe avec la probabilité la plus élevée
                        class_index = max(probabilities, key=probabilities.get)
                        class_info = CLASSES.get(class_index, {"name": "Classe inconnue", "emoji": "❓"})
                        
                        # Suppression de la barre de progression
                        progress_bar.empty()
                        
                        # Affichage du résultat avec une jolie boîte
                        st.markdown(
                            f"""
                            <div class="prediction-box">
                                <h3>{class_info['emoji']} Résultat de la classification</h3>
                                <p><strong>Commentaire :</strong> {commentaire}</p>
                                <p><strong>Catégorie principale :</strong> {class_info['name']} <small>(probabilité: {probabilities[class_index]:.2f})</small></p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Affichage du graphique des probabilités
                        show_probability_chart(probabilities)
                        
                        # Ajout à l'historique
                        add_result_to_history(commentaire, probabilities, lang_code)
                
                except Exception as e:
                    st.error(f"⚠️ Erreur: {str(e)}")
                    logging.error(f"Erreur lors de la prédiction: {str(e)}")
    
    # Séparateur visuel
    st.markdown("---")
    
    # Affichage des résultats précédents
    display_results()
    
    # Pied de page
    st.markdown(
        """
        <div style="text-align: center; margin-top: 40px; color: #7F8C8D;">
        <p>© 2025 - Application de classification automatique de commentaires</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    setup_logging()
    main()