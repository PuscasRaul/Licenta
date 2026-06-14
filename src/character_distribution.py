#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

data = {np.str_('0'): 743, np.str_('1'): 697, np.str_('2'): 693,
        np.str_('3'): 737, np.str_('4'): 626, np.str_('5'): 1042,
        np.str_('6'): 744, np.str_('7'): 708, np.str_('8'): 632,
        np.str_('9'): 775, np.str_('A'): 1056, np.str_('B'): 569,
        np.str_('C'): 589, np.str_('D'): 470, np.str_('E'): 588,
        np.str_('F'): 603, np.str_('G'): 513, np.str_('H'): 852,
        np.str_('I'): 330, np.str_('J'): 387, np.str_('K'): 617,
        np.str_('L'): 603, np.str_('M'): 701, np.str_('N'): 593,
        np.str_('O'): 440, np.str_('P'): 426, np.str_('Q'): 300,
        np.str_('R'): 652, np.str_('S'): 474, np.str_('T'): 669,
        np.str_('U'): 380, np.str_('V'): 370, np.str_('W'): 361,
        np.str_('X'): 350, np.str_('Y'): 346, np.str_('Z'): 321}

keys = data.keys()
values = data.values()

plt.bar(keys, values)
plt.xlabel('Caractere')
plt.ylabel('Număr de intrări')
plt.title('Distribuția caracterelor în setul de date')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
