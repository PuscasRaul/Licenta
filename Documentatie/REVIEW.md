# Revizuire lucrare de licență

Generat: 2026-06-01. Review pe versiunea curentă a capitolelor (după rescriere). Analiză independentă a fișierelor `.tex`, raportată la structura din comentariile `main.tex:47-57`.

> Notă de scop: se evaluează **doar ce este scris**. Capitolele 5 (parțial) și 6 (placeholder), Abstractul și subcapitolele de testare/experimente sunt marcate neutru în §6, fără a fi tratate ca defecte.

---

## 1. Evaluare de ansamblu

Față de versiunea anterioară, lucrarea s-a îmbunătățit substanțial: introducerea are acum cele două părți cerute de coordonator (ce/de ce/cum + structura pe capitole), label-urile sunt unice și consecvente (`chap:localizare`, `chap:segmentare`, `chap:recunoastere`, `chap:implementare`, `chap:concluzii`), iar toate cheile `\cite` au corespondent în `references.bib`. Problemele din review-ul precedent (citări lipsă, label duplicat, dedup Hough) sunt rezolvate.

Structura *general → specific* pe primele trei capitole funcționează și respectă cerința. Capitolele teoretice sunt bine scrise, coerente și cu un nivel de detaliu peste medie pentru o licență.

**Răspuns la întrebarea „e prea multă teorie?":** În ansamblu, **nu** — nivelul e adecvat pentru o lucrare al cărei titlu accentuează „clasificatori statistici" și a cărei contribuție e tocmai *clasic + SVM în loc de deep learning*. Există însă **un singur punct de dezechilibru clar** (teoria VC din Cap. 4, vezi §2.1) și **două locuri cu adâncime ușor disproporționată** pentru metode care nu sunt folosite mai departe (vezi §2.2). Restul e justificat.

Observație transversală: capitolele 2 și 3 sunt aproape pur *survey de literatură* (fiecare metodă = o lucrare + rata ei de acuratețe), iar justificarea alegerilor proprii e amânată la Cap. 5. Este o decizie legitimă (oglindește chiar structura din `SOA_Review.pdf` pe care o citezi), dar înseamnă că cititorul parcurge ~2 capitole fără să afle ce ai ales și de ce. O frază de legătură la finalul fiecărui capitol teoretic („dintre acestea, în sistemul propus s-a optat pentru ... din motivele detaliate în Cap. 5") ar lega survey-ul de contribuție fără a muta justificarea.

---

## 2. Echilibru și teorie nelalocul ei

### 2.1. Capitolul 4 — teoria VC e cea mai grea parte a întregii lucrări (dezechilibru real)
**Fișier:** `chapters/chapter4.tex:91-114`

Subsubsecțiunea *Teoria VC și clasificatori cu marjă maximală* introduce: dimensiunea VC, mărginirea generalizării prin **inegalitatea completă** cu 8 termeni itemizați, principiul minimizării riscului structural. Este, de departe, cel mai dens pasaj teoretic din lucrare — depășește nivelul aplicativ al restului.

- **Defensabil parțial:** alegerea SVM peste ANN e justificată ulterior (`chapter4.tex:156`) tocmai prin garanțiile de generalizare, deci conceptul de capacitate VC *are* un rol în argument.
- **Recomandare:** păstrează intuiția (dimensiune VC = capacitate, compromis fidelitate/generalizare, marja maximală ca mod de a o controla) și **comprimă inegalitatea cu 8 termeni** la o formă mai scurtă sau mută detalierea termenilor într-o notă/anexă. În forma actuală, o comisie poate întreba „de ce e nevoie de derivarea completă a marginii într-o lucrare aplicativă?".

**Dezechilibru SVM vs. ANN în același capitol:** SVM primește ~2 pagini cu 3 subsubsecțiuni (spațiul trăsăturilor, kernel trick, teoria VC, marjă moale), ANN primește 3 paragrafe plate (`chapter4.tex:144-156`). Direcția e corectă (SVM e cel folosit), dar contrastul e abrupt. Fie scurtezi SVM (vezi sus), fie precizezi explicit la începutul secțiunii *Clasificatori statistici* că SVM e tratat în detaliu fiindcă e cel adoptat, ANN doar comparativ.

### 2.2. Adâncime ușor disproporționată pentru metode neutilizate
- **`chapter3.tex:75-79` — ecuația eikonală + level-set (Fast Marching).** Formalizarea matematică (ecuația eikonală, funcția de viteză) e relativ grea pentru o metodă care e doar una dintre cele patru prezentate și care **nu e folosită** în sistem. Sunt doar ~2 paragrafe, deci acceptabil, dar e candidatul nr. 2 la trimming dacă vrei să reduci teoria.
- **`chapter2.tex:118-126` — harta de densitate ponderată (5 ecuații).** Setul de 5 ecuații pentru `bremananth2005robust` e prezentat fără ca rezultatul lor (cum se obțin efectiv regiunile candidate) să fie închis: `chapter2.tex:128` spune doar „putem identifica posibilele regiuni" fără a explica *cum*. Ori reduci ecuațiile la ideea de principiu, ori închizi explicația.

### 2.3. Template matching — proporție bună
`chapter4.tex:7-49` tratează template matching echilibrat: principiu, limitări, exemplu concret (Hamming, normalizare la 40×40). Lungime potrivită pentru o metodă prezentată ca *baseline* clasic. OK.

---

## 3. Erori factuale și de logică (de prioritate mare)

### 3.1. Atribuire geografică greșită
**`chapter5.tex:51`** — „`\cite{shi2005automatic}`, care exploatează caracteristicile cromatice ale plăcuțelor **taiwaneze**". Eroare: `shi2005automatic` tratează plăcuțe **din China** (caractere albe pe albastru / negre pe galben — cf. `chapter2.tex:17`). `chang2004automatic` e cel cu Taiwanul. Inversează atribuirea. Plus „spațiul autohton al insulei Taiwan" — *autohton* față de cine? Reformulează (ex. „specifice plăcuțelor din Taiwan").

### 3.2. Propoziție întreruptă — definiție lipsă
**`chapter4.tex:116`** — „...mărginită superior de raportul $R^2/\gamma^2$, unde $\gamma$ reprezintă." Fraza se termină brusc: **nu se spune ce este $\gamma$** (marja). De completat („...unde $\gamma$ reprezintă marja geometrică a hiperplanului separator").

### 3.3. Relație logică inversată perceptron ↔ SVM/ANN
**`chapter4.tex:55`** — „ambele își au rădăcinile în algoritmul perceptronului, însă abordările matematice care **precedă** acest algoritm diferă fundamental." SVM și ANN **nu preced** perceptronul — îl *succed* / pornesc de la el. Înlocuiește „precedă" cu „pornesc de la" / „dezvoltă" / „succedă".

### 3.4. Atribuire ambiguă a acurateței de 98%
**`chapter2.tex:140`** — „...mult mai robuste ... comparat atât cu metodele bazate pe contururi cât și cele bazate pe culoare `\cite{zunino2002vector}` care au atins o acuratețe de 98%, nu pot fi folosite...". Nu e clar **ce** a atins 98% (metoda pe textură? zunino?) și `zunino2002vector` pare lipit greșit de „metodele bazate pe culoare". Reformulează fraza — e și prea lungă (un singur enunț pe 4 rânduri).

### 3.5. Încadrarea negativă a operatorului Sobel
**`chapter2.tex:85`** — „Datorită complexității ridicate a operatorului Sobel, în `\cite{VEDA}`...". Sobel tocmai fusese prezentat ca soluție bună (`luo2009efficient`, 47 ms). Tranziția sună contradictoriu. Reformulează: „Deși Sobel oferă rezultate bune, costul său de calcul a motivat..." sau similar.

---

## 4. Erori LaTeX / referințe

- **`chapter4.tex:36`** — „Formula acestei distanțe este descrisă de **Ecuația 4.1**." Numărul e scris manual (hardcoded). Adaugă `\label{eq:hamming}` la ecuație și folosește `\eqref{eq:hamming}` — altfel, dacă inserezi o ecuație înainte, numărul devine greșit.
- **`chapter2.tex:48` și `:93`** — `47~\text{ms}`, `15~\text{ms}`, `130~\text{ms}`. Funcționează, dar `\text{ms}` în math mode e neconvențional pentru o unitate. Preferabil `47~ms` (text) sau pachetul `siunitx` cu `\SI{47}{\milli\second}`. Consecvent în toată lucrarea.
- **`chapter2.tex:49-53`** — figura `figures/luo2209` are typo în numele fișierului (probabil `luo2009`). Verifică să existe fișierul, altfel imaginea nu compilează.
- **`chapter2.tex:61-83`** — figura cu 5 subfiguri (input/greyscale/median/sobel/open) **nu are `\caption` general** și nici `\label`. Adaugă un caption la nivel de `figure` și un label, ca să o poți referenția.
- **`main.tex:47-57`** — blocul de comentarii cu instrucțiunile de structurare e încă în fișier. Util ca ghidaj, dar **de șters înainte de predare**.

---

## 5. Typo-uri și exprimare (pe capitole)

### Introducere (`chapter1_introduction.tex`)
- `:36` — „În comparație **existente**" → lipsește un cuvânt: „În comparație **cu soluțiile** existente".
- `:17` — „plăcuțe uzuale --două litere --două cifre --trei litere" — formatarea cu liniuțe e confuză. Folosește o paranteză descriptivă: „(format standard: două litere, grup de cifre, trei litere)".

### Capitolul 2 (`chapter2.tex`)
- `:48` — „Rezultatul ... este **binarizată**" → „**binarizat**" (acord cu „rezultatul").
- `:48` — „algoritmi de prag precum metoda histogramei, **grad4** etc." — *grad4* pare termen incomplet/greșit. Probabil voiai un nume de algoritm de prag (Otsu?). Verifică.
- `:87` — „imaginea greyscale **îî** este aplicat" → „**îi**" (dublu î).
- `:55` — „atenuarea zgomotului presupune crearea unei funcții care capturează ... **lasând**" → „**lăsând**".
- `:57` — „asupra căreia să continue cu **urmatoarele**" → „**următoarele**".
- `:108` — „Implementarea **transformatii** Hough" → „**transformării**".

### Capitolul 3 (`chapter3.tex`)
- `:15` — „filtrarea regiunilor **identificare** de CCA" → „**identificate**".
- Capitol în general curat și bine scris.

### Capitolul 4 (`chapter4.tex`)
- `:5` — „există două variante" — frază trunchiată, fără punct și fără a anunța care sunt cele două variante (urmează direct `\section`). Închide cu „: potrivirea de șabloane și clasificatorii statistici." sau o frază introductivă.
- `:13` — „fiind **extrem de** sensibilă" — registru informal; „deosebit de" / „foarte" mai potrivit academic.
- `:49` — „nrows și ncols **sunt sunt** numărul" — „sunt" dublat.
- `:53` — „utilizate în **creare** numerelor" → „**crearea**"; tot acolo „o **rezistența** ridicată" → „**rezistență**".
- `:59` — „regresiei/ **clasficării**" → „**clasificării**".

### Capitolul 5 (`chapter5.tex`) — partea scrisă
- `:11` — concentrare de typo-uri într-o frază: „a **fos** implementat" → „fost"; „**gradulu** ridicat" → „gradul"; „sistemelelor de tip embedded" → „sistemelor"; „**detrimenul**" → „detrimentul".
- `:13` — „specifice **dezvolării** aplicației" → „dezvoltării".
- `:51` — paragraful e bine plasat aici (decizie de aplicație, nu teorie ✔), dar se termină **fără punct** și conține eroarea factuală din §3.1.

---

## 6. Părți încă nescrise (semnalate neutru, nu ca defecte)

Doar pentru evidență, conform structurii din `main.tex:54-56`:
- `chapter5.tex:45,47,49,52` — subsubsecțiuni/subsecțiuni cu heading gol (Sistem embedded, Aplicație demo, Fluxul de procesare, Nivelul aplicație).
- Subcapitol de testare (metrici) + subcapitol de experimente — anunțate în intro (`chapter1_introduction.tex:55-57`), de scris.
- `chapter6_conclusions.tex` — placeholder „Concluzii ...".
- Abstract (`main.tex:41`) — neimplementat.
- Numele aplicației: intro folosește deja `\textsc{IORA}` (`chapter1_introduction.tex:55`), dar titlul Cap. 5 e încă generic „Implementarea sistemului". Când completezi, aliniază titlul la numele aplicației (cerință `main.tex:54`).

---

## 7. Prioritizare

| # | Acțiune | Efort | Impact |
|---|---------|-------|--------|
| 1 | Eroare factuală shi2005/Taiwan (§3.1) + frază $\gamma$ neterminată (§3.2) + logica perceptron (§3.3) | mic | mare (corectitudine) |
| 2 | Reformulare frază ambiguă 98%/zunino (§3.4) + tranziție Sobel (§3.5) | mic | mediu |
| 3 | Comprimarea teoriei VC / reechilibrare SVM↔ANN (§2.1) | mediu | mare (echilibru) |
| 4 | `\eqref` la ecuația Hamming + caption/label figuri + nume fișier luo (§4) | mic | mediu (compilare) |
| 5 | Frază de legătură survey→contribuție la finalul Cap. 2 și 3 (§1) | mic | mediu |
| 6 | Typo-uri (§5) — pe parcurs | mic | cosmetic |
| 7 | Ștergere comentarii `main.tex` înainte de predare | mic | obligatoriu |
