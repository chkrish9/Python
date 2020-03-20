"Question 2"


def main():
    inptstring = input("Please enter the string to manipulate:")
    optstring = string_alternative(inptstring)
    print(optstring)


def string_alternative(inptstring):
    optstring=""
    for i in range(len(inptstring)):
        if i % 2 == 0:
            optstring += inptstring[i]
    return optstring


main()