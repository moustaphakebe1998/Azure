import streamlit as st
import urllib.request
import json
import ssl
import os
import pandas as pd
import logging

# Assurer que le dossier de logs existe
os.makedirs('logs', exist_ok=True)

# Configuration des logs
logging.basicConfig(
    filename='logs/app.log',  # Chemin vers le fichier de log
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_unverified_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

# Légende des classes
classes = {
    0: "Activités commerciales et professionnelles",
    1: "Arbres, végétaux et animaux",
    2: "Autos, motos, vélos, trottinettes...",
    3: "Autos, motos, vélos...",
    4: "Eau",
    5: "Graffitis, tags, affiches et autocollants",
    6: "Mobiliers urbains",
    7: "Objets abandonnés",
    8: "Propreté",
    9: "Voirie et espace public"
}

# Titre de l'application
st.title("Classification de commentaire")

# Liste pour stocker les résultats
if 'results_list' not in st.session_state:
    st.session_state['results_list'] = []

# Entrée utilisateur
commentaire = st.text_input("Commentaire", "Des Autocollants sur mur")

if st.button("Prédire"):
    logging.info(f"Commentaire reçu : {commentaire}")

    data = {
        "Inputs": {
            "data": [
                {
                    "commentaire_usager": commentaire,
                }
            ]
        }
    }

    body = str.encode(json.dumps(data))
    url = 'http://98.66.231.209:80/api/v1/service/dmrprd/score'
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        result_json = json.loads(result)

        # Afficher la réponse JSON brute dans la barre latérale
        with st.sidebar:
            if st.checkbox("Afficher la réponse JSON brute"):
                st.write("Réponse JSON brute:")
                st.write(result_json)  # Affiche la réponse JSON brute
                st.write("")  # Espace pour la lisibilité

            # Afficher les classes disponibles
            st.write("Classes disponibles:")
            for index, label in classes.items():
                st.write(f"{index}: {label}")

        # Vérification de la structure de la réponse
        if isinstance(result_json, dict) and "Results" in result_json:
            results = result_json["Results"]

            # Gérer si "Results" est une liste d'entiers (ex: [5])
            if isinstance(results, list) and len(results) > 0:
                class_index = results[0]  # Accéder au premier élément de la liste
                if isinstance(class_index, int):
                    class_label = classes.get(class_index, "Classe inconnue")
                    st.write(f"Classe prédite : {class_label}")
                    logging.info(f"Classe prédite : {class_label}")
                    
                    # Créer un DataFrame avec le commentaire et la classe prédite
                    st.session_state['results_list'].append({'Commentaire': commentaire, 'Classe prédite': class_label})
                    
                    # Créer un DataFrame avec tous les résultats
                    df = pd.DataFrame(st.session_state['results_list'])
                    
                    # Afficher le DataFrame
                    st.write(df)
                    
                    # Convertir le DataFrame en CSV
                    csv = df.to_csv(index=False, sep=";")
                    
                    # Créer un bouton de téléchargement pour le CSV
                    st.download_button(
                        label="Télécharger le CSV",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv'
                    )
                else:
                    st.error("L'élément de résultat n'est pas un entier valide.")
                    logging.error("L'élément de résultat n'est pas un entier valide.")

            # Gérer si "Results" est un dictionnaire (ex: {"0": 5})
            elif isinstance(results, dict):
                for key, class_index in results.items():  # Itérer sur les paires clé-valeur
                    if isinstance(class_index, int):
                        class_label = classes.get(class_index, "Classe inconnue")
                        st.write(f"Classe prédite : {class_label}")
                        logging.info(f"Classe prédite : {class_label}")
                        
                        # Ajouter le résultat à la liste des résultats
                        st.session_state['results_list'].append({'Commentaire': commentaire, 'Classe prédite': class_label})
                        
                        # Créer un DataFrame avec tous les résultats
                        df = pd.DataFrame(st.session_state['results_list'])
                        
                        # Afficher le DataFrame
                        st.write(df)
                        
                        # Convertir le DataFrame en CSV
                        csv = df.to_csv(index=False, sep=";")
                        
                        # Créer un bouton de téléchargement pour le CSV
                        st.download_button(
                            label="Télécharger le CSV",
                            data=csv,
                            file_name='predictions.csv',
                            mime='text/csv'
                        )
                    else:
                        st.error("L'élément de résultat n'est pas un entier valide.")
                        logging.error("L'élément de résultat n'est pas un entier valide.")
            else:
                st.error("La structure des résultats de l'API est incorrecte.")
                logging.error("La structure des résultats de l'API est incorrecte.")

    except urllib.error.HTTPError as error:
        st.error(f"La requête a échoué avec le code de statut: {error.code}")
        st.error(error.read().decode("utf8", 'ignore'))
        logging.error(f"La requête a échoué avec le code de statut: {error.code}")
    except Exception as e:
        st.error(f"Une erreur s'est produite: {str(e)}")
        logging.error(f"Une erreur s'est produite: {str(e)}")