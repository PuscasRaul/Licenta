# Capitolul 5 — Structură propusă (ghid de scriere)

> Notă de lucru pentru redactarea `chapters/chapter5.tex`. Conține structura
> detaliată, ce intră în fiecare (sub)capitol și legăturile cu codul din `src/`.
> Titlul capitolului ar trebui să fie mai specific (cu numele aplicației), nu
> generic „Implementarea sistemului".

---

## Verdict pe structura curentă

Ce e deja bun și se păstrează:
- Împărțirea în 3 module (Input / Processing / Application) cu polimorfism la
  margini și pipeline închis la mijloc — fidel codului
  (`ICaptureManager` + `CameraCaptureManager`, pipeline izolat, FastAPI ca strat
  de aplicație).
- `Tehnologii` împărțit în Nucleu / Nivel aplicație — corespunde decuplării
  dependențelor (sklearn/opencv/numpy vs fastapi/websockets).
- Paralela embedded vs. demo la modulele extensibile.

Ce lipsește (din comentariile din `main.tex`, liniile 54–56 + 47):
1. Titlu specific cu numele aplicației.
2. Subcapitol de **testare → metrici**.
3. Subcapitol de **experimente**.
4. Documentație tip ISS: **diagrame, actori**.
5. Subcapitol despre **construirea setului de date** (3 scripturi:
   `parse_characters.py`, `generate_dataset.py`, `label_dataset.py`).

Dezechilibru de corectat: „Fluxul de procesare" e inima lucrării (~600 linii de
viziune artificială cu decizii de design reale), trebuie să fie cel mai dezvoltat
subcapitol, nu un singur paragraf.

---

## Structura detaliată

### 5.1 Tehnologii  *(deja există, ok)*
- **5.1.1 Nucleu** — OpenCV, numpy, scikit-learn, matplotlib. Argumentul
  minimizării dependențelor pentru portabilitate / embedded.
- **5.1.2 Nivelul de aplicație** — fastapi, websockets (opționale, depind de
  contextul aplicației).

### 5.2 Arhitectura sistemului  *(NOU — aici intră diagramele ISS)*
- Diagramă de componente: cele 3 module + fluxul de date între ele.
- Diagramă use-case + actori (operator, sistem barieră, serviciu de notificare).
- Decizia de design: interfețe abstracte la margini, pipeline închis la mijloc
  (mută aici argumentul din introducerea capitolului).

### 5.3 Modulul de intrare (Input)
- **5.3.1 Interfața `ICaptureManager`** — semnătura `get_frame(self) -> Frame`,
  obiectul `Frame` (image_data, timestamp, metadata). Lipsă de dependențe.
- **5.3.2 Implementare: sistem embedded** — captură din cameră
  (`CameraCaptureManager`, OpenCV `VideoCapture`, gestiunea celui mai recent
  cadru / buffer size).
- **5.3.3 Implementare: aplicația demo.**

> ⚠ Sincronizare cod–text: interfața din lucrare zice `get_frame`, dar
> `CameraCaptureManager` are `get_latest_frame`. Aliniază numele înainte de a
> cita semnături.

### 5.4 Fluxul de procesare  *(cel mai dezvoltat subcapitol)*

#### 5.4.1 Considerații preliminare
- Paragraful existent despre excluderea segmentării bazate pe culoare
  (color-based segmentation) + motivația abordării pe muchii/morfologie.
- Cele 3 etape clasice: localizare → segmentare → recunoaștere.

#### 5.4.2 Localizarea plăcuței (`LPExtraction`)
- Preprocesare: grayscale, filtru median.
- Accentuarea muchiilor: Sobel pe orizontală.
- Binarizare Otsu (inversată) — de ce inversată (caracterele întunecate devin
  prim-plan alb pentru `findContours`).
- Morfologie: opening + dilatare orizontală scalată cu lățimea imaginii.
- Extragere contururi + filtrare (aspect ratio, arie minimă).
- Reasamblarea fragmentelor: union-find (`_merge_split_regions`).
  **Decizie de design**: de ce aici și nu în morfologie (risc de a uni plăcuța
  cu bara/grila).
- Rankarea candidaților prin densitatea de muchii + întoarcerea `top_k`.
- Corecția înclinării (deskew) prin transformata Hough.

#### 5.4.3 Segmentarea caracterelor (`CharacterSegmentation`)  *(METODĂ NOUĂ)*

> **Schimbare de abordare.** Se trece de la varianta actuală
> (prag adaptiv gaussian → contururi → selecție pe deviația standard) la o
> abordare bazată pe **analiza proiecțiilor**, combinată cu un `_score` care
> exploatează faptul că un număr de înmatriculare are un număr cunoscut de
> caractere (obiecte).

Conținut propus:

1. **Motivația trecerii la proiecții.** Limitele metodei pe contururi:
   sensibilitate la zgomot/conectarea caracterelor, dependență de praguri pe
   deviația standard, caractere care fuzionează/se rup. Proiecțiile dau o
   reprezentare 1D robustă a distribuției pixelilor de prim-plan.

2. **Pipeline-ul pe proiecții:**
   - Binarizare (Otsu inversat) → imagine de prim-plan alb.
   - **Proiecția orizontală** (sumă pe linii) pentru a izola banda/benzile de
     text (plăcuțe pe un rând vs. moto pe două rânduri).
   - **Proiecția verticală** (sumă pe coloane) pe fiecare bandă de text pentru a
     găsi granițele caracterelor: caracterele = vârfuri (peaks), spațiile dintre
     ele = văi (valleys).
   - Extragerea regiunilor-vârf cu prag relativ la maximul histogramei
     (`extract_peak_regions`, deja schițat în cod) → bounding-box-uri pe
     caractere.

3. **`_score` bazat pe proiecția verticală (criteriul principal).**
   Scopul: a alege, dintre candidații `top_k` de plăcuță, segmentarea cea mai
   plauzibilă. Ideea-cheie: **numărul de obiecte (vârfuri) este cunoscut a
   priori** pentru o plăcuță validă. Componente ale scorului:
   - **Numărul de vârfuri vs. numărul așteptat** de caractere (penalizare pe
     `|n − n_ideal|`; pt. RO tipic 6–7 simboluri).
   - **Regularitatea lățimilor** vârfurilor (caractere de lățime comparabilă →
     deviație standard mică).
   - **Regularitatea spațiilor** (văile) dintre vârfuri — spațiere uniformă.
   - **Contrast vârf/vale** — vârfuri bine separate (văi adânci) indică o
     segmentare curată.

4. **Analiza criss-cross (cross-counting) — alternativă/complement.**
   Pe baza paper-ului citit: pentru fiecare coloană (și/sau rând) se numără
   **tranzițiile alb↔negru** (crossings). O coloană din interiorul unui caracter
   are multe tranziții; o coloană dintr-un spațiu între caractere are puține.
   Profilul de crossings înlocuiește/întărește proiecția simplă de pixeli și
   este mai robust la grosimea trăsăturii (stroke width). Poate fi folosit:
   - ca semnal de segmentare (tăieturi în coloanele cu crossings minime), și/sau
   - ca termen în `_score` (consistența numărului de crossings per caracter).

   > 🔎 De adăugat: citarea paper-ului cu criss-cross / crossing-count
   > segmentation în `references.bib`.

5. **Selecția candidatului câștigător.** Legătura `top_k` între 5.4.2 și 5.4.3:
   extragerea întoarce mai mulți candidați de plăcuță, segmentarea rulează pe
   fiecare și păstrează setul de caractere cu `_score` maxim (recuperare din
   erori de localizare în etapa de segmentare — decizie de design de evidențiat).

> 🧹 Curățenie: `extract_using_projections` / `_handle_text_line` din codul
> actual sunt incomplete (au un bug — `y` nedefinit, slicing greșit). Acestea
> devin baza noii implementări; de rescris și de prezentat ca pipeline activ,
> nu ca „abordare explorată".

#### 5.4.4 Recunoașterea caracterelor (clasificator SVM)
- Reprezentarea caracterului: redimensionare 28×28, normalizare [0,1], flatten.
- Kernel RBF, hiperparametrii `C`, `gamma` — legătură cu capitolul teoretic SVM.
- Filtrarea claselor cu prea puține mostre (`MIN_SAMPLES_PER_CLASS`).

### 5.5 Construirea setului de date  *(NOU)*
- **5.5.1 Sursa externă adnotată** — `parse_characters.py`, format Pascal
  VOC/XML, extragere caractere pe bounding-box.
- **5.5.2 Auto-generarea prin pipeline** — `generate_dataset.py`: rulează
  localizarea + segmentarea pe imagini și salvează caracterele tăiate.
- **5.5.3 Etichetarea manuală asistată** — `label_dataset.py`: interfață OpenCV
  pentru etichetare la tastă, mutarea în foldere pe clasă, marcarea „bad".
- **5.5.4 Filtrarea și igienizarea claselor** — prag minim de mostre, eliminare
  etichete non-ASCII / non-alfanumerice.

> ⚠ Sincronizare cod–text: `generate_dataset.py` importă încă numele vechi
> (`DetectionPipeline` din `src.license_plate_extraction`). De aliniat cu
> `LPExtraction` / `CharacterSegmentation` înainte de a cita.

### 5.6 Nivelul de aplicație
- **5.6.1 Interfața de logică / acțiuni declanșate** (polimorfism).
- **5.6.2 Implementare: sistem embedded** — barieră auto, notificări e-mail,
  acționare hardware.
- **5.6.3 Implementare: aplicația demo** — FastAPI + WebSocket
  (streaming rezultate către client).

### 5.7 Testare și metrici  *(din comentariile tale)*
- Metodologie: split stratificat 80/20, `random_state` fix pentru
  reproductibilitate.
- Metrici: acuratețe globală, `classification_report` (precision/recall/F1 per
  clasă), matrice de confuzie.
- Eventual: metrici separate pe etape (rata de localizare corectă, rata de
  segmentare corectă) vs. metrica end-to-end (plăcuța întreagă corect citită).

### 5.8 Experimente  *(din comentariile tale)*
- Studii de caz pe factorii de reziliență din introducere: lumină variabilă,
  formate diferite de plăcuță, reziduuri care obscurează caracterele.
- Ablații: efectul `top_k`, al deskew-ului, al `_merge_split_regions`.
- Comparație directă **segmentare pe contururi (veche) vs. proiecții + `_score`
  (nouă)** — justifică schimbarea de abordare cu numere.

---

## Decizii de design de evidențiat explicit în text
- **Cuplajul `top_k`** între localizare și segmentare (recuperare din erori).
- **`_merge_split_regions` cu union-find** și de ce reasamblarea se face după
  contururi, nu în morfologie.
- **`_score` bazat pe cunoștința a priori** a numărului de caractere — leagă
  segmentarea de o ipoteză de domeniu, nu de praguri arbitrare.

## TODO de sincronizare cod ↔ lucrare (înainte de a cita semnături)
- [ ] `get_frame` vs `get_latest_frame` în modulul de intrare.
- [ ] `DetectionPipeline` / `src.license_plate_extraction` → `LPExtraction`.
- [ ] Rescrierea segmentării pe proiecții + `_score` (înlocuiește metoda pe std).
- [ ] Citarea paper-ului criss-cross în `references.bib`.


Loaded 20957 samples across 36 classes
Per-class counts: {np.str_('0'): 743, np.str_('1'): 697, np.str_('2'): 693, np.str_('3'): 737, np.str_('4'): 626, np.str_('5'): 1042, np.str_('6'): 744, np.str_('7'): 708, np.str_('8'): 632, np.str_('9'): 775, np.str_('A'): 1056, np.str_('B'): 569, np.str_('C'): 589, np.str_('D'): 470, np.str_('E'): 588, np.str_('F'): 603, np.str_('G'): 513, np.str_('H'): 852, np.str_('I'): 330, np.str_('J'): 387, np.str_('K'): 617, np.str_('L'): 603, np.str_('M'): 701, np.str_('N'): 593, np.str_('O'): 440, np.str_('P'): 426, np.str_('Q'): 300, np.str_('R'): 652, np.str_('S'): 474, np.str_('T'): 669, np.str_('U'): 380, np.str_('V'): 370, np.str_('W'): 361, np.str_('X'): 350, np.str_('Y'): 346, np.str_('Z'): 321}

Accuracy: 0.963

Classification report:
              precision    recall  f1-score   support

           0       0.95      0.93      0.94       149
           1       0.96      0.98      0.97       139
           2       0.97      0.98      0.97       139
           3       0.99      0.99      0.99       147
           4       0.98      0.99      0.99       125
           5       0.99      1.00      1.00       209
           6       0.97      0.98      0.98       149
           7       0.97      0.99      0.98       142
           8       0.98      0.98      0.98       126
           9       0.97      0.98      0.98       155
           A       0.97      1.00      0.99       211
           B       0.99      0.96      0.97       114
           C       0.93      0.97      0.95       118
           D       0.98      0.95      0.96        94
           E       0.96      0.97      0.97       118
           F       0.98      0.97      0.97       121
           G       0.97      0.90      0.93       103
           H       0.96      0.97      0.96       170
           I       0.82      0.82      0.82        66
           J       0.95      0.94      0.94        77
           K       0.95      0.99      0.97       123
           L       0.99      0.94      0.97       121
           M       0.96      0.94      0.95       140
           N       0.97      0.97      0.97       119
           O       0.85      0.89      0.87        88
           P       0.93      0.94      0.94        85
           Q       0.98      0.93      0.96        60
           R       0.96      0.98      0.97       130
           S       0.99      0.98      0.98        95
           T       0.96      0.98      0.97       134
           U       0.93      0.93      0.93        76
           V       0.96      0.92      0.94        74
           W       0.96      0.94      0.95        72
           X       0.98      0.93      0.96        70
           Y       0.95      0.90      0.93        69
           Z       0.93      0.97      0.95        64

    accuracy                           0.96      4192
   macro avg       0.96      0.95      0.96      4192
weighted avg       0.96      0.96      0.96      4192


Confusion matrix (rows = true, cols = predicted):
      0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R   S   T   U   V   W   X   Y   Z
  0 139   0   0   0   0   0   1   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   7   0   0   0   0   0   1   0   0   0   0   0
  1   0 136   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1
  2   0   0 136   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   1
  3   0   0   0 145   0   1   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  4   0   0   1   0 124   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  5   0   0   0   0   0 209   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  6   0   0   0   0   0   0 146   0   2   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  7   0   0   0   0   0   0   0 140   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   1
  8   0   0   0   0   0   0   1   0 123   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0
  9   0   1   0   0   0   0   0   0   1 152   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0
  A   0   0   0   0   0   0   0   0   0   0 211   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  B   0   0   0   0   0   0   0   0   0   0   0 109   0   0   1   0   0   0   0   0   0   0   0   0   0   1   1   1   0   0   1   0   0   0   0   0
  C   0   0   0   0   0   0   0   0   0   0   0   0 115   0   0   0   1   0   0   0   0   0   0   0   1   1   0   0   0   0   0   0   0   0   0   0
  D   0   0   0   0   0   0   0   0   0   0   1   0   0  89   0   1   0   0   0   1   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0
  E   0   0   0   0   0   0   0   0   0   0   0   0   1   0 115   0   0   0   0   1   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0
  F   0   0   0   0   0   0   1   1   0   0   0   0   0   0   0 117   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   1   0   0
  G   0   0   0   0   1   0   0   0   0   2   0   0   5   0   0   0  93   0   0   1   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0
  H   0   1   0   0   1   1   0   0   0   0   0   0   0   0   1   0   0 165   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  I   0   3   0   1   0   0   0   0   0   0   1   0   0   0   0   0   0   0  54   0   0   1   0   0   0   0   0   1   0   3   1   0   0   0   1   0
  J   0   0   0   0   0   0   0   0   0   1   0   0   0   1   0   0   0   1   1  72   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0
  K   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 122   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0
  L   0   1   1   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   1   0   0 114   1   0   0   0   0   1   0   0   0   0   0   0   0   1
  M   0   0   0   0   0   0   0   1   0   0   0   0   0   1   0   0   0   4   2   0   0   0 132   0   0   0   0   0   0   0   0   0   0   0   0   0
  N   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2 116   0   0   0   0   0   1   0   0   0   0   0   0
  O   7   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0  78   0   0   0   1   0   0   0   0   0   0   0
  P   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   1   1   1   0   0   0   0   0   0  80   0   1   0   0   0   0   0   0   0   0
  Q   1   0   0   0   0   0   0   0   0   1   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0  56   0   0   0   1   0   0   0   0   0
  R   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   1   0 127   0   0   0   0   0   0   0   1
  S   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0  93   0   0   0   0   0   0   0
  T   0   0   0   0   0   0   0   1   0   0   0   0   0   0   1   0   0   0   1   0   0   0   0   0   0   0   0   0   0 131   0   0   0   0   0   0
  U   0   0   0   0   0   0   0   0   0   0   0   0   1   0   1   0   0   0   1   0   0   0   1   0   0   0   0   0   0   1  71   0   0   0   0   0
  V   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   1   0   0   2   0   0   0   0   0   0   0  68   1   0   1   0
  W   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   1   0   0   0   0   0   0   0   0   0   1   1  68   0   0   0
  X   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   3   0   0   0   0   0   0   0   0   0   0   0   0  65   1   0
  Y   0   0   1   0   0   0   0   1   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   2   0   0   0   0   0   2   0   0  62   0
  Z   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0  62
