import re, sys

data = open(sys.argv[1], "r").readlines()

for row in data:
    _,_,text, _,lang = row.strip().split("|")
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    for w in words:
        print("__label__"+lang+" "+w)
        
