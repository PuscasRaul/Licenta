import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Definim datele tale sub formă de text (exact cum le-ai dat mai sus)
data_str = """
      0   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R   S   T   U   V   W   X   Y   Z
  0 139   0   0   0   0   0   1   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   7   0   0   0   0   0   1   0   0   0   0   0
  1   0 136   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1
  2   0   0 136   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   1
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
"""

# 2. Parsăm textul într-o matrice de date
lines = data_str.strip().split('\n')
columns = lines[0].split()
data = []
index = []

for line in lines[1:]:
    parts = line.split()
    index.append(parts[0]) # Primul element e eticheta randului
    data.append([int(x) for x in parts[1:]]) # Restul sunt numerele

# 3. Creăm un tabel (DataFrame) cu Pandas
df_cm = pd.DataFrame(data, index=index, columns=columns)

# 4. Desenăm graficul (Heatmap)
plt.figure(figsize=(16, 12)) # Setăm o dimensiune generoasă
sns.heatmap(df_cm,
            annot=False,    # Oprește scrierea cifrelor în fiecare căsuță (ar fi prea aglomerat)
            cmap='Blues',   # Schema de culori
            fmt='g',
            linewidths=0.5, # Delimitare mai clară
            cbar_kws={'label': 'Frecvență'})

plt.title('Matrice de Confuzie', fontsize=18, pad=20)
plt.ylabel('Eticheta Reală (True Label)', fontsize=14)
plt.xlabel('Eticheta Prezisă (Predicted Label)', fontsize=14)

# Rotim etichetele pentru a fi mai vizibile
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()

# Salvăm rezultatul ca imagine (sau poți folosi plt.show() ca să-l vizualizezi pe ecran)
plt.savefig('matrice_confuzie.png', dpi=300)
print("Graficul a fost salvat cu succes!")
