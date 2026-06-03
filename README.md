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
   macro avg       0.79      0.86      0.82       570
weighted avg       0.91      0.90      0.90       570

## Andra saker vi har testat

### Patrik
### Vi testade att inkludera kordinaterna för människorna på de bilderna där de förekommer. Resultet blev snarlikt det vi redan fått.

### Viktor
### Vi testade att köra modellen med utökad syntetisk data utan någon förbättring. De syntetiska bilderna hade andra miljöer och andra vinklar samt enbart människor och inga bilar.