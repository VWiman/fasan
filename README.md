# FASAN
# Fältbaserad AI för Sökning Av Nödställda


# DATASET URL
https://www.kaggle.com/datasets/nikolasgegenava/sard-search-and-rescue

## Resultat

### Vi fick ett bra resultat men kan se på GRAD-CAM att den tror att även bilar är människor.

Classification report

                  precision    recall  f1-score   support

        no human       0.61      0.81      0.70        84
           human       0.97      0.91      0.94       486

        accuracy                           0.90       570
    macro avg          0.79      0.86      0.82       570
    weighted avg       0.91      0.90      0.90       570

## Andra saker vi har testat

### Träning med koordinater
Undersökte om inkludering av de bounding box-koordinater ([x, y, w, h]) i träningen kunde förbättra modellens förmåga att känna igen människor (multi-task learning). Dessa koordinater fanns med i datasetet.

**Tillvägagångssätt:**
Vi började med att använda koordinater som "facit" under träning. Detta tvingade modellen att lära sig bildinnehåll bättre, vilket resulterade i en träffsäkerhet på 93% för Human.
Vid nästkommande körning önskades en Yolo-liknande modell som också skulle prediktera var människan befann sig. Modellens resultat var mycket gott i att avgöra om det fanns en människa på bilden eller inte, men box-prediktionen visade sig dock bristfällig vid visuell inspektion. Därefter förenklades modellen till en binär klassificerare och optimerades för prestanda genom *Focal Loss* och tröskelvärdesjusteringar för att balansera hög *Human Recall* mot en rimlig nivå av falsklarm.

**Slutsats:**
Genom att använda koordinater som "facit" under träning lyckades vi lära modellen att se människor mycket bättre än med den ursprungliga baslinjen.

## Justeringar:
*   Lokaliseringsträning var det enskilt viktigaste steget för att tvinga modellen att lära sig se objektet "människa" korrekt.
*   Focal Loss i kombination med tröskelvärdesjustering gav den bästa balansen för praktisk tillämpning i Search and Rescue, där hög Human Recall är prioriterat utan att dränkas i falsklarm.

**Resultat från utforskad modell (Threshold Tuning):**

                  precision    recall  f1-score   support

        no human       0.65      0.83      0.73        84
           human       0.97      0.92      0.95       486

        accuracy                           0.91       570
    macro avg          0.81      0.88      0.84       570
    weighted avg       0.92      0.91      0.91       570


**Slutlig reflektion:**
Att utforska lokalisering och avancerade loss-funktioner var en mycket intressant väg. När vi jämför den slutgiltiga utforskade modellen med vår ursprungliga version kan vi konstatera att förbättringarna var marginella, vilket tyder på att den ursprungliga arkitekturen redan var väl anpassad för uppgiften.

Resultat för den ursprungliga modellen:

                  precision    recall  f1-score   support

        no human       0.69      0.77      0.73        84
           human       0.96      0.94      0.95       486

        accuracy                           0.92       570
    macro avg          0.83      0.86      0.84       570
    weighted avg       0.92      0.92      0.92       570




### Viktor
### Vi testade att köra modellen med utökad syntetisk data utan någon förbättring. De syntetiska bilderna hade andra miljöer och andra vinklar samt enbart människor och inga bilar.
