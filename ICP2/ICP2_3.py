"Question 3"
fileLoc = input("Enter the file location:")
infile = open(fileLoc,'r')
line = infile.readline()
words = {}
while line != "":
    wordlist = line.split(" ")
    for i in range(len(wordlist)):
        if wordlist[i] in words.keys():
            words[wordlist[i]] = words[wordlist[i]]+1
        else:
            words[wordlist[i]] = 1

    line = infile.readline()

for x in words:
    print (x, ':', words[x])