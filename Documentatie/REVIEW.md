# Revizuire lucrare de licență — listă de modificări

Generat: 2026-05-28. Status: review după feedback coordonator + check independent al fișierelor `.tex` și `references.bib`.

---

## 1. Feedback explicit de la coordonator

### 1.1. Reorganizarea introducerii în două părți
**Fișier:** `chapters/chapter1_introduction.tex`

Coordinatorul a cerut explicit (confirmat și de comentariile din `main.tex:46-56`) ca introducerea să aibă:

- **Partea I (≈1 pagină):** *ce fac, de ce fac, cum fac, de ce e important / ce aduce în plus*
- **Partea II (≈1/2 pagină):** * structura lucrării pe capitole *

**Stare actuală:**
- Partea I există parțial, dar amestecată cu taxonomie tehnică și aplicații (ordine confuză).
- Partea II **lipsește complet** — nu există niciun paragraf care să anunțe ce conține fiecare capitol.

**De făcut:**
1. Restructurare Partea I în ordinea: context → *ce fac* (sistem ALPR modular, tehnici clasice + SVM) → *de ce* (cerința parcării din RO, diversitatea formatelor, constrângerea real-time) → *de ce e bun* (reziliență la condiții variabile + optimizare).
2. Mută paragraful de taxonomie multi-etapă vs. single-stage (`chapter1_introduction.tex:26`) în Capitolul 2 — e prea tehnic pentru intro.
3. Mută `\subsection{Contextul aplicației}` în corpul Părții I, fără heading separat (sau promovează-l la `\section`).
4. Adaugă Partea II — un paragraf de tip *"Lucrarea este organizată în șase capitole..."* care descrie pe scurt fiecare capitol.
5. Subliniază în Partea II că **primele 3 capitole sunt teoretice** (general → specific), iar Capitolul 5 e contribuția practică (cerință explicită din comentariu `main.tex:51`).

### 1.2. Generalizarea Capitolului 2
**Fișier:** `chapters/chapter2.tex`

Titlul e deja *"Localizarea obiectelor"* — bun. Conținutul însă rămâne 100% despre plăcuțe.

**De făcut:**
1. `chapter2.tex:5` — reformulează deschiderea: *"Detectarea unui obiect de interes într-o imagine..."* în loc de *"Detectarea corectă și precisă a plăcuței de înmatriculare (LP)..."*. Introdu LP-ul doar ca exemplu concret.
2. `chapter2.tex:12` — exemplul *"trecerile sunt în general de la alb la negru, roșu la alb, verde la roșu"* — reformulează general: *"obiectele cu contur bine definit prezintă tranziții bruște de intensitate la marginile lor"*.
3. **Mută în Capitolul 5** justificările care țin de decizii de aplicație, nu de teorie:
   - `chapter2.tex:7` — paragraful despre excluderea color-based segmentation (e **duplicat** cu `chapter1_introduction.tex:28`)
   - `chapter2.tex:9` — paragraful despre excluderea metodelor bazate pe textură
4. În subsecțiunile Sobel/VEDA/Hough — păstrează prezentarea teoretică, dar folosește LP-ul doar ca exemplu/aplicație.
5. Structura ideală: *teorie generală → exemple din literatură ALPR* (nu invers, cum e acum).

---

## 2. Probleme structurale majore

### 2.1. Capitolul 5 incomplet
**Fișier:** `chapters/chapter5.tex`

Subcapitole goale (doar heading, fără conținut):
- `chapter5.tex:45` — `\subsubsection{Sistem embedded}`
- `chapter5.tex:47` — `\subsubsection{Aplicație demo}`
- `chapter5.tex:49` — `\subsection{Fluxul de procesare}`
- `chapter5.tex:51` — `\subsection{Nivelul aplicație}`

**Lipsesc (conform comentariu `main.tex:54-56`):**
- Subcapitol de testare cu metrici
- Subcapitol de experimente
- Redenumirea capitolului cu numele aplicației (în loc de generic *"Implementarea sistemului"*)

### 2.2. Concluzii inexistente
**Fișier:** `chapters/chapter6_conclusions.tex`

Conține literalmente *"Concluzii ..."* (un placeholder). De scris integral.

### 2.3. Abstract neimplementat
**Fișier:** `main.tex:40-62`

`ABSTRACT` e doar o linie + două paragrafe care repetă conținut din introducere. Trebuie scris ca rezumat real al întregii lucrări, **inclusiv rezultate** și concluzii.

### 2.4. Conținut duplicat între intro și Cap. 2
Paragraful despre excluderea color-based segmentation (referințe `shi2005automatic` și `chang2004automatic`) apare aproape identic în:
- `chapter1_introduction.tex:28`
- `chapter2.tex:7`

Păstrează **o singură instanță** — recomandat în Cap. 5 (decizie de aplicație), cu o trimitere scurtă în intro.

---

## 3. Erori de LaTeX / numerotare

### 3.1. Label duplicat
Două capitole diferite au același `\label{chap:ch3}`:
- `chapter4.tex:1` → `\chapter{Recunoașterea caracterelor}~\label{chap:ch3}`
- `chapter5.tex:1` → `\chapter{Implementarea sistemului}\label{chap:ch3}`

Va produce warning LaTeX *"multiply defined labels"*. Recomandare:
- `chapter4.tex` → `\label{chap:recunoastere}`
- `chapter5.tex` → `\label{chap:implementare}`

### 3.2. Label inconsistent cu numărul capitolului
- `chapter2.tex:2` → `\label{chap:ch1}` în Capitolul 2. Redenumește în `\label{chap:localizare}`.

### 3.3. Math mode lipsă
- `chapter2.tex:57` → *"kernel de 2\times4"* — în text liber, `\times` nu se randează. Schimbă în `$2 \times 4$`.

### 3.4. Comentarii orfane în `main.tex`
- `main.tex:46-56` — bloc mare de comentarii cu instrucțiuni de structurare. Sunt utile ca ghidaj, dar **trebuie șterse înainte de predare**.
- `main.tex:69-70` și `main.tex:78-79` — `\addcontentsline` comentat. Verifică dacă cuprinsul afișează corect *Introducere* și *Concluzii* — dacă nu, decomentează.

---

## 4. Citări lipsă din `references.bib`

Aceste chei sunt folosite în `.tex` dar **NU există** în `references.bib` (vor produce `[?]` în PDF):

| Cheie               | Folosită în       | Context                                           |
|---------------------|-------------------|---------------------------------------------------|
| `paliy2004approach` | `chapter3.tex:28` | Segmentare prin redimensionare la mărime standard |
| `sarfraz2003saudi`  | `chapter4.tex:10` | Template matching cu distanță Hamming             |

**De făcut:** caută referințele și adaugă-le în `references.bib`. Probabil:
- Paliy, I. et al. — *"Approach to recognition of license plate numbers using neural networks"* (2004, IEEE IJCNN)
- Sarfraz, M. et al. — *"Saudi Arabian license plate recognition system"* (2003, Int. Conf. Geometric Modeling)

### 4.1. Citare duplicată în `references.bib`
`HoughTransformation` (`references.bib:11-18`) și `duan2005building` (`references.bib:183-190`) sunt **același articol** (Duan, Du, Phuoc, Hoang — *"Building an automatic vehicle license plate recognition system"*, RIVF 2005).

Folosit cu chei diferite în:
- `chapter2.tex:67` → `\cite{HoughTransformation}`
- `chapter3.tex:20` → `\cite{duan2005building}`

**De făcut:** păstrează o singură cheie (recomandat `duan2005building`) și actualizează `chapter2.tex:67`. Sau invers.

---

## 5. Typo-uri și greșeli de exprimare

### Capitolul 1 (Introducere)
- `chapter1_introduction.tex:6` — *"este are aplicații"* → *"are aplicații"*
- `chapter1_introduction.tex:6` — *"este cu atât mai mare"* nu se leagă gramatical de subiectul *"crearea și dezvoltarea unui sistem"*. Reformulează.
- `chapter1_introduction.tex:8` — *"Vaiații"* → *"Variații"*
- `chapter1_introduction.tex:26` — *"singură etapă"* → *"singură etapă"* (formă inconsistentă cu *"multi-etapă"*); folosește *"o singură etapă"*
- `chapter1_introduction.tex:28` — *"sistem decontrol"* → *"sistem de control"* (spațiu lipsă)

### Capitolul 2
- `chapter2.tex:5` — *"acuratețea"* nu apare aici, dar pe `chapter2.tex:9` — *"acurateței"* → *"acurateții"*
- `chapter2.tex:14` — *"Rezultatul intermediar generat de operatorul Sobel este binarizată"* → *"...este binarizat"* (acord)
- `chapter2.tex:14` — *"grad4"* — pare un termen incomplet. Probabil voiai *"Otsu"* sau alt nume de algoritm de prag. Verifică.
- `chapter2.tex:14` — *"un timp mediu de procesare de 47ms"* — adaugă spațiu: *"47 ms"* (norma SI).
- `chapter2.tex:21` — *"lasând"* → *"lăsând"*
- `chapter2.tex:23` — *"asupra căreia să continue cu urmatoarele"* → *"urmatoarele"* → *"următoarele"*
- `chapter2.tex:25` — *"central fiind setat ca alb"* — în descrierea eroziunii corecte, eroziunea micșorează regiunile albe (foreground). Verifică logica — pare inversată cu dilatarea.
- `chapter2.tex:51` — *"Datorită complexității ridicate a operatorului Sobel"* — context: dar tocmai ai prezentat Sobel ca soluție. Reformulează: *"Cu toate că Sobel oferă rezultate bune, complexitatea sa..."*.
- `chapter2.tex:57` — *"2\times4"* → `$2 \times 4$` (vezi §3.3)
- `chapter2.tex:59` — *"atine"* → *"atinge"*
- `chapter2.tex:59` — *"15ms"* / *"130ms"* — adaugă spațiu: *"15 ms"* / *"130 ms"*
- `chapter2.tex:67` — *"15~\textdegree"* — `\textdegree` cere `\gensymb` ✓ (ai pachetul). OK.
- `chapter2.tex:74` — *"transformatii"* → *"transformării"*
- `chapter2.tex:74` — *"30~\textdegree comparat cu 15~\textdegree"* — adaugă punct/virgulă după unitate pentru lizibilitate.

### Capitolul 3
- `chapter3.tex:13` — *"identificare"* → *"identificate"* (acord cu *"regiunilor"*)
- `chapter3.tex:20` — *"lp-ului"* — folosește forma extinsă *"numărului de înmatriculare"* sau definește abrevierea LP la prima apariție și păstreaz-o consecvent (cu majuscule).
- `chapter3.tex:22` — *"Această abordare este extrem de utilă"* — *"extrem de"* e expresie informală; recomandat *"deosebit de utilă"*.
- `chapter3.tex:24` — *"de început și de final a fiecărui caracter"* → *"ale fiecărui caracter"* (acord plural).
- `chapter3.tex:27` — *"significant"* → *"semnificativ"* (anglicism).
- `chapter3.tex:28` — *"un modul de extrage"* → *"un modul de extragere"*.
- `chapter3.tex:30` — *"matrița"* — neclar în context; probabil *"șablonul"* sau *"matricea"*.

### Capitolul 4
- `chapter4.tex:14` — propoziție trunchiată: *"diferiți algorimti care calculează distanțe: , Mahalanobis"* — lipsește text înainte de virgulă. Plus typo *"algorimti"* → *"algoritmi"*.
- `chapter4.tex:12` și `chapter4.tex:14` repetă același mesaj cu cuvinte diferite. Comasează.
- `chapter4.tex:17` — *"robuste. Acestea prezintă o rezistența"* → *"rezistență"* (acord nedeterminat).
- `chapter4.tex:17` — *"variațiile de luminozitate, dimensiuni sau fonturi variate utilizate în creare numerelor"* → *"crearea numerelor"*.
- `chapter4.tex:19` — *"abordările matematice care precedă acest algoritm"* — formulare ambiguă. SVM nu *precede* perceptronul; vrei *"succede"* sau *"generalizează"*.
- `chapter4.tex:19` — *"clasficării"* → *"clasificării"*.
- `chapter4.tex:22` — lipsește spațiu după punct înainte de citare: *"~\cite{vapnik2015uniform}.Fundamentele"* → adaugă spațiu după punct.
- `chapter4.tex:24` — *"așa că"* — folosit corect, dar verifică tonul academic; *"prin urmare"* / *"așadar"* mai potrivit.
- `chapter4.tex:30` — *"într-o dimensiune diferită"* → *"într-un spațiu de dimensiune diferită"* (clarificare).

### Capitolul 5
- `chapter5.tex:11` — *"a fos implementat"* → *"a fost implementat"*.
- `chapter5.tex:11` — *"gradulu"* → *"gradul"*.
- `chapter5.tex:11` — *"sistemelelor"* → *"sistemelor"*.
- `chapter5.tex:11` — *"detrimenul"* → *"detrimentul"*.
- `chapter5.tex:11` — *"dezvolării"* (apare în subcap. *Tehnologii*) → *"dezvoltării"*.
- `chapter5.tex:8` — *"a fiecărui modul"* — verifică acord; aici e corect (genitiv neutru).
- `chapter5.tex:39` — interfața din verbatim are doar 1 metodă `get_frame` — fă un comentariu/precizare dacă e completă sau e doar exemplu.

### `main.tex`
- `main.tex:25` — *"Informatică Română"* — verifică terminologia oficială. La UBB Cluj specializarea se numește *"Informatică, linia de studiu în limba română"* sau similar. Confirmă cu coordinator.
- `main.tex:26` — titlul lucrării conține *"vedere artificială"* — în literatura RO recentă se folosește mai des *"viziune artificială"* / *"viziune computațională"*. Sugerez să verifici cu coordinator.
- `main.tex:60` — *"segmenatarea"* → *"segmentarea"*.
- `main.tex:60` — *"recunoașterea captează"* → *"de recunoaștere captează"* (apare ca *"Sistemul de recunoașterea captează"*).
- `main.tex:62` — *"sistemlui"* → *"sistemului"*.
- `main.tex:62` — *"in această"* → *"în această"*.

---

## 6. Recomandări stilistice (opțional)

- Folosește consecvent **LP** sau **NÎ** sau forma extinsă *"plăcuța de înmatriculare"* / *"numărul de înmatriculare"*. Acum amesteci toate trei.
- Cifrele cu unități: adaugă spațiu non-secabil între număr și unitate (`47~ms`, `15~\textdegree`).
- `\paragraph{}` folosit ca simulator de paragraf nou în `chapter1_introduction.tex` și `main.tex` — nu e necesar; lasă LaTeX să gestioneze paragrafele cu linie albă.
- Note de tehnoredactare: în Cap. 2 figurile au caption-uri foarte scurte (*"Imaginea Sobel"*) — extinde la *"Rezultatul aplicării operatorului Sobel asupra imaginii filtrate"*.
- Cap. 2, figura `luo2209` — typo în numele fișierului (probabil `luo2009`). Verifică `figures/`.

---

## 7. Ordinea recomandată de rezolvare

1. **Citările lipsă** (paliy2004approach, sarfraz2003saudi) + dedup HoughTransformation — efort mic, impact vizibil în PDF -> DONE
2. **Label-uri duplicate** — efort minim, evită warnings -> DONE, cred
3. **Restructurare introducere** (Partea I + Partea II) — efort mediu, cerință directă coordonator
4. **Generalizare Cap. 2** — efort mediu, cerință directă coordonator
5. **Eliminare duplicate** intro/Cap.2 (color-based segmentation)
6. **Typo-uri** — pe parcurs, în timp ce editezi fiecare capitol
7. **Completare Cap. 5** (subcapitole goale, testare, experimente)
8. **Scriere Cap. 6** Concluzii
9. **Scriere Abstract real** (după ce ai rezultatele)
10. **Curățare comentarii din `main.tex`**
