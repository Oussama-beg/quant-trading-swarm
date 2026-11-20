# Student 1 — Market Analysis Agent

Agent d'analyse de marché du **Projet 2 (Swarm de Trading Quantitatif & Gestion du Risque)**.
LSTM entraîné sur 20 actions (dataset *Stock Market Dataset — Jackson Crow*) pour prédire la direction du prix à J+1 (`Target_Up`).

## Contenu
| Fichier | Rôle |
|---|---|
| `data_utils.py` | Préparation des données commune à l'équipe (split chronologique 70/15/15 par action, features, cibles) |
| `Student1_Market_Agent.ipynb` | Entraînement et évaluation du LSTM, génération de la sortie de l'agent |
| `market_agent_output.json` | Sortie de l'agent pour Student 3 (une entrée par action) |
| `lstm_market_agent.keras` | Modèle LSTM entraîné |
| `scaler_market.pkl` | StandardScaler ajusté sur le train |

## Données et features
- 20 actions (`TICKERS` dans `data_utils.py`), `SEQ_LEN = 20` jours.
- `TECH_COLS` : Return, Close_MA5_Ratio, Close_MA20_Ratio, MA5_MA20_Ratio, Daily_Spread, **RSI_14, MACD_Norm, MACD_Hist_Norm** (ajouts Student 1).
- `RISK_COLS` : Volatility_7d, Volatility_21d, Daily_Spread, Return_Lag1.
- Cibles : `Target_Up` (hausse à J+1), `Is_Safe` (rendement J+1 > -1,5 %).
- Scaler ajusté uniquement sur le train ; séquences construites à l'intérieur de chaque action.

## Lancer
1. Télécharger le dataset Kaggle *Stock Market Dataset* (Jackson Crow) et placer les CSV dans `data/stocks/` (ex. `data/stocks/AAPL.csv`). Sur Kaggle : *Add Input*.
2. `pip install -r requirements.txt`
3. Exécuter `Student1_Market_Agent.ipynb` (le dataset n'est pas versionné dans ce dépôt).

## Résultats (jeu de test, 27 432 séquences)
| Métrique | Valeur |
|---|---|
| Accuracy | 0.483 (baseline classe majoritaire : 0.522) |
| AUC | 0.502 |
| F1 | 0.060 |

- Le modèle prédit « baisse » dans environ 97 % des cas : il a appris la prior du train (47,6 % de hausses).
- Aucune action ne montre de signal exploitable (AUC entre 0.47 et 0.54).
- **Conclusion** : la direction à J+1 est quasi imprévisible avec ces features, ce qui est cohérent avec l'hypothèse de marché efficient.

## Sortie pour Student 3 (`market_agent_output.json`)
Par ticker : `predicted_direction`, `probability_up`, `RSI_14`, `MACD_Norm`, `Volatility_7d`, `Volatility_21d`, `date`.

Recommandation d'usage : traiter `probability_up` comme un signal faible et s'appuyer surtout sur les indicateurs de risque (volatilités, RSI, MACD) pour le filtrage du risque.

## Limites
- Split fait par action (convention d'équipe) : les périodes de test diffèrent d'une action à l'autre ; des dates de coupure communes seraient plus strictes.
- Les données s'arrêtent en avril 2020 (le test inclut la crise COVID).
- Pistes : pondération des classes / seuil de décision, prédiction de la volatilité, validation walk-forward.
