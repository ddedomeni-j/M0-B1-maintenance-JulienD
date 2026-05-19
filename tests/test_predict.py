"""Tests fonctionnels de l'endpoint /predict.

Permet de tester la sortie dans dans le cas valide et dans le cas d'un type de machine invalide.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_predict_with_correct_input_returns_200() -> None:
    """L'endpoint /predict doit répondre 200 OK si l'entrée est correcte."""
    with TestClient(app) as client:
        data_ok = {"age_machine_jours": 1500,
                   "derniere_maintenance_jours": 45,
                   "nb_incidents_3_mois": 2,
                   "pression_moyenne": 7.8,
                   "temperature_moyenne": 68.5,
                   "type_machine": "compresseur",
                   "vibration_moyenne": 3.2}
        
        response = client.post("/predict", json=data_ok)
        assert response.status_code == 200
        body = response.json()
        sum_proba = sum(body["probabilites"].values())
        assert abs(sum_proba-1.0) < 0.01
        
        # A voir : modèle déterministe?
        assert body["criticite"] == "basse"
        assert body["probabilites"]["basse"]==0.97
        assert body["probabilites"]["moyenne"]==0.03
        assert body["probabilites"]["haute"]==0


def test_predict_with_wrong_machine_type_returns_error_422() -> None:
    """L'endpoint /predict doit retourner une erreur 422  si le type de machine est incorrect."""
    
    with TestClient(app) as client:
        data_ko = {"age_machine_jours": 1500,
                   "derniere_maintenance_jours": 45,
                   "nb_incidents_3_mois": 2,
                   "pression_moyenne": 7.8,
                   "temperature_moyenne": 68.5,
                   "type_machine": "presse_agrume",
                   "vibration_moyenne": 3.2}
        
        response = client.post("/predict", json=data_ko)
        body = response.json()
        assert response.status_code == 422