# FASAN — Fältbaserad AI för Sökning Av Nödställda

Ett binärt CNN som avgör om det finns en människa i en flygbild eller inte. Projektet är byggt som ett påhittat uppdrag: att hitta nödställda personer i terräng, oavsett om de syns tydligt eller är delvis dolda. Prioriteringen är hög *recall* — hellre ett falsklarm än en missad människa.

Skapat som grupparbete i kursen *Tillämpad AI: maskininlärning och deep learning*.

## Innehåll

- [Översikt](#översikt)
- [Dataset](#dataset)
- [Installation](#installation)
- [Användning](#användning)
- [Projektstruktur](#projektstruktur)
- [Modell](#modell)
- [Konfiguration](#konfiguration)
- [Resultat](#resultat)
- [Andra spår vi testat](#andra-spår-vi-testat)
- [Tester](#tester)
- [Notebook och presentation](#notebook-och-presentation)

## Översikt

Uppgiften formuleras som binär klassificering: *finns det en människa i bilden eller inte?* Indata är flygbilder med YOLO-etiketter. En tom etikettfil tolkas som `no human`, en fil med minst en bounding box tolkas som `human`. Datasetet är tydligt obalanserat (cirka 86 % bilder med människa), vilket hanteras med class weights och dataaugmentering.

Hela arbetsflödet — rensning, dataladdning, EDA, modellbygge, träning, utvärdering och Grad-CAM — körs från en sammanhållen pipeline. Samma kod importeras av projektets notebook, så inget dupliceras.

## Dataset

SARD – Search and Rescue, hämtat från Kaggle:

https://www.kaggle.com/datasets/nikolasgegenava/sard-search-and-rescue

Ladda ner datasetet och placera det i mappen `dataset/` med undermapparna `train/`, `valid/` och `test/`, var och en med `images/` och `labels/` enligt YOLO-formatet (se `data.yaml`). Fördelningen är 4 041 träningsbilder, 1 144 valideringsbilder och 570 testbilder (5 755 totalt).

## Installation

Projektet kräver Python 3.10+ och TensorFlow 2.18.

```bash
pip install -r requirements.txt
```

Huvudberoenden: `tensorflow`, `numpy`, `scikit-learn`, `pillow`, `matplotlib`, `tqdm` och `pytest` för testerna.

## Användning

Kör hela pipelinen — rensning, EDA, träning, utvärdering och Grad-CAM — med:

```bash
python pipeline.py
```

Enskilda steg kan köras separat:

```bash
python eda.py        # Skapar EDA-grafer i output/eda
python grad_cam.py   # Grad-CAM-heatmaps och IoU-utvärdering
```

Alla resultat sparas under `output/` (checkpoints, EDA-grafer, träningsgrafer och Grad-CAM-bilder).

## Projektstruktur

```
search-and-rescue/
├── pipeline.py        # Kör hela flödet från rådata till Grad-CAM
├── config.py          # All konfiguration (bildstorlek, hyperparametrar, sökvägar)
├── clean.py           # Städar bort tidigare output
├── clean_data.py      # Rensar och kontrollerar datasetet
├── load_data.py       # Läser bilder och YOLO-etiketter, bygger dataset
├── eda.py             # Explorativ dataanalys
├── graphs.py          # Gemensamma grafifunktioner
├── model.py           # CNN-arkitektur
├── training.py        # Träning och utvärdering
├── grad_cam.py        # Grad-CAM-heatmaps och IoU-mätning
├── data.yaml          # Datasetkonfiguration (YOLO-format)
├── dataset/           # train / valid / test (laddas ner separat)
├── output/            # Genererade resultat (checkpoints, grafer, rapporter)
├── tests/             # Pytest-svit
├── FASAN_notebook.ipynb     # Genomgång av hela flödet
└── FASAN_presentation.pptx  # Presentationsdeck
```

## Modell

Ett CNN för binär klassificering, byggt i Keras:

- **Input:** 640 × 640 × 3
- **Faltningsblock:** filterstorlekar `[32, 64, 128, 256]`, varje block med två Conv2D följt av MaxPooling
- **GlobalMaxPooling2D**
- **Dense:** `[256, 256]` med L2-regularisering och dropout 0.25
- **Output:** Dense(1) med sigmoid → sannolikhet
- **Optimerare:** AdamW, learning rate 1e-4
- **Loss:** binary crossentropy

Träningen använder class weights i läge `balanced`, dataaugmentering (flip, lätt rotation, ljus, kontrast och färg — endast på träningsdata), EarlyStopping på `val_loss` samt ReduceLROnPlateau.

## Konfiguration

All konfiguration ligger samlad i `config.py` och styr hela pipelinen — bildstorlek, batchstorlek, antal epoker, learning rate, augmenteringsfaktorer, class weight-läge, trösklar och utdatamappar. Ändra parametrar där snarare än i de enskilda modulerna.

## Resultat

Modellen hittar de flesta människorna. Genom att balansera datasetet och prioritera hög recall blir avvägningen rätt för Search and Rescue. Grad-CAM visar dock att exakt lokalisering är svagare, och att modellen ibland aktiverar på fordon.

```
                precision    recall  f1-score   support

    no human       0.61      0.81      0.70        84
       human       0.97      0.91      0.94       486

    accuracy                           0.90       570
   macro avg       0.79      0.86      0.82       570
weighted avg       0.91      0.90      0.90       570
```

Grad-CAM-utvärderingen mot YOLO-boxarna ger ett medel-IoU på cirka 0.15, och omkring 15 % av bilderna lokaliseras korrekt (IoU ≥ 0.30). Modellen avgör alltså säkert *om* en människa finns, men pekar inte alltid ut exakt *var*.

## Andra spår vi testat

### Träning med koordinater (multi-task)

Vi undersökte om bounding box-koordinaterna `[x, y, w, h]` i datasetet kunde förbättra modellen genom multi-task learning. Genom att använda koordinaterna som facit under träningen tvingades modellen att lära sig bildinnehållet bättre, vilket gav 93 % träffsäkerhet för `human` — marginellt bättre än baslinjen.

Ett försök med en YOLO-liknande modell som också skulle prediktera var människan befann sig fungerade väl på *om*-frågan, men box-prediktionen var bristfällig vid visuell inspektion. Modellen förenklades därför tillbaka till en binär klassificerare och optimerades med Focal Loss och tröskeljustering för att balansera hög Human Recall mot en rimlig nivå av falsklarm:

```
                precision    recall  f1-score   support

    no human       0.65      0.83      0.73        84
       human       0.97      0.92      0.95       486

    accuracy                           0.91       570
   macro avg       0.81      0.88      0.84       570
weighted avg       0.92      0.91      0.91       570
```

Lokaliseringsträningen var det enskilt viktigaste steget för att tvinga modellen att se objektet "människa" korrekt, och Focal Loss med tröskeljustering gav den bästa praktiska balansen. När den slutgiltiga utforskade modellen jämförs med baslinjen är skillnaderna ändå marginella, vilket tyder på att den ursprungliga arkitekturen redan var väl anpassad för uppgiften.

### Utökat dataset

Vi testade att träna med utökad syntetisk data — bilder med andra miljöer och vinklar, med enbart människor och inga fordon — utan någon mätbar förbättring.

## Tester

Projektet har en pytest-svit som validerar dataladdning, datarensning, modellbygge, träning och grafer:

```bash
pytest
```

## Notebook och presentation

`FASAN_notebook.ipynb` går igenom hela flödet i ordning och visar sparade resultat utan att träna om modellen. `FASAN_presentation.pptx` sammanfattar projektet på tio slides.
