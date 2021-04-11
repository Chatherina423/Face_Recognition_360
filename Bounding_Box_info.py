import os
import glob

files = glob.glob('/Users/suhyeonyoo/Downloads/archive/Annotation/*.txt')
print(files)

for f in files:
    file = open(f)
    line = file.readline()
    print(line)