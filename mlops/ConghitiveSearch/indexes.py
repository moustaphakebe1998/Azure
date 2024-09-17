import requests
import json

# Configuration Azure Cognitive Search
search_endpoint = "https://ia-searchk....search.windows.net"  # URL de votre service Azure Cognitive Search
admin_key = "fok2jLVK4MdHuem78RTDd5jmWMaR..."  # Clé API administrateur pour Azure Cognitive Search

# Définition de l'index en JSON
index_definition = {
    "name": "kebe-index",
    "fields": [
        {
            "name": "id",
            "type": "Edm.String",
            "key": True,
            "filterable": False,
            "sortable": False,
            "facetable": False
            
        },
        {
            "name": "content",
            "type": "Edm.String",
            "searchable": True,
            "filterable": False,
            "sortable": False,
            "facetable": False,
            "analyzer": "standard.lucene"
        },
        {
            "name": "embedding",
            "type": "Collection(Edm.Single)",
            "searchable": False,
            "filterable": False,
            "sortable": False,
            "facetable": False
        }
    ]
}

# En-têtes pour la requête HTTP
headers = {
    "Content-Type": "application/json",
    "api-key": admin_key
}

# URL pour la création de l'index
url = f"{search_endpoint}/indexes?api-version=2021-04-30-Preview"

# Requête HTTP pour créer l'index
response = requests.post(url, headers=headers, data=json.dumps(index_definition))

if response.status_code == 201:
    print("Index créé avec succès.")
else:
    print(f"Erreur lors de la création de l'index: {response.status_code}, {response.text}")
