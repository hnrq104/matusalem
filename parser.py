import csv
import re
import sys

pattern = r"(?P<book>[1-3]?\s?\w+(?:\s\w+){0,2})\s(?P<chapter>\d+):(?P<versicle>\d+)\t(?P<text>.+)"
bible_lines = open('dados/kjv.txt','r').read().splitlines()

with open('dados/bible.csv','w') as csvfile:
    fieldnames = ['book','chapter','versicle','text']
    writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
    writer.writeheader()
    for line in bible_lines[2:]: #skipping the first lines
        str = re.sub(r"[\[\]\];]", "", line)
        m = re.match(pattern,str)
        if m:
            writer.writerow(m.groupdict())
        else:
            print(f"erro: ao parsear {line}!",file=sys.stderr)
