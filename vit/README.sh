#Author: M. El Aabaribaoune

vit_downscaling/
│
├── train.py              → Lancement de l'entraînement
├── evaluate.py           → Inférence & diagnostics
│
├── config.yaml           → Configuration globale (reproductibilité)
├── config.py             → Wrapper YAML → Python
│
├── data_loading.py       → Lecture ERA5 / MSWEP + masques
├── preprocessing.py     → Normalisation & tenseurs
│
├── training.py           → Boucle d’entraînement générique
├── evaluation.py         → Évaluation & sauvegardes NetCDF
│
├── vit_arch.py           → Architecture Vision Transformer
│   └── (cnn_arch.py, unet_arch.py, rf_model.py …)
│
├── utils.py              → Fonctions utilitaires (plots, IO, paths)
│
└── results/
    ├── train/
    └── test/
