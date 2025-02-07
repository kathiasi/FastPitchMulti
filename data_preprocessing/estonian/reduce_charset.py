import os, sys
import locale

def custom_sort_key(char):
    return (char.islower(), locale.strxfrm(char))

    
# Set locale for Estonian sorting
locale.setlocale(locale.LC_ALL, "et_EE.UTF-8")

metadata = open(sys.argv[1]).readlines()
"""
all_chars = {}
for l in metadata:
    _,_, prompt, _,_ = l.split('|')
    for c in list(prompt):
        if c in all_chars:
            all_chars[c]+=1
        else:
            all_chars[c] = 1

    #return (locale.strxfrm(char.upper()), locale.strxfrm(char))
    
sorted_chars = sorted(all_chars.items(), key=lambda item: item[1])
#for char, count in sorted_chars:
#    print(char, count)
"""

accent_map = {
    "á": "a", "à": "a", "â": "a", "ā": "a",  # Keep "ä"
    "é": "e", "è": "e", "ê": "e", "ë": "e", "ẽ": "e", "ē": "e",
    "í": "i", "ì": "i", "î": "i", "ï": "i", "ĩ": "i", "ī": "i",
    "ó": "o", "ò": "o", "ô": "o", "ø": "o", "ō": "o",  # Keep "õ" and "ö"
    "ú": "u", "ù": "u", "û": "u", "ũ": "u", "ū": "u",  # Keep "ü"
    "É":"E", "Ž":"Z", "–": "-"  #others
}

# Generate the translation table
accent_table = str.maketrans(accent_map)

remove_table = str.maketrans(dict.fromkeys("<>&", " "))

quote_table = str.maketrans(dict.fromkeys("“”„‘’'«»", "\""))  

all_chars = {}

for l in metadata:
    cols = l.strip().split('|')
    translated = cols[2].translate(accent_table)
    translated = translated.translate(quote_table)
    translated = translated.translate(remove_table)
    translated = translated.replace("\x00", "")
    translated = " ".join(translated.split())

    for c in list(translated):
        if c in all_chars:
            all_chars[c]+=1
        else:
            all_chars[c] = 1

    cols[2] = translated
    print("|".join(cols))


#sorted_chars = sorted(all_chars.items(), key=lambda item: item[1])
#print([t[0] for t in sorted_chars])
print (sorted(all_chars.keys(),key=custom_sort_key))
#for char, count in sorted_chars:
#    print(char, count)
