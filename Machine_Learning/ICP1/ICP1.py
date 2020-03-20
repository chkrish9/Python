"Question 1"
"""
1)	Division Operation : In python 2 x version after division operation the result is rounded off to the floor value where as in python 3x the division operation maintains precision and provides the exact decimal value.
2)	Xrange : Python 2x has xrange function which gives us iterator object as  result whereas in python 3x version there is no xrange function instead it has range function which returns numbers.
3)	Print :  Python 2x has no parenthesis 
4)	Error Handling : There is a small change in error handling in both versions. In python 3.x, ‘as’ keyword is required. 
"""

"Question 2"
inpt = input("Enter the string you want to manipulate: ")
inpt = inpt[0:-2]
print(inpt)
inpt = inpt[::-1]
print(inpt)

num1 = float(input("Enter the a number: "))
num2 = float(input("Enter the a number: "))
option = int(input("1. Add \n2. Sub \n3. Mul \n4. Div \n"))
if option == 1:
    print("Addition of two numbers: ", num1+num2)
elif option == 2:
    print("Subtraction of two numbers: ", num1-num2)
elif option == 3:
    print("Multiplication of two numbers: ", num1 * num2)
elif option == 4:
    print("Division of two numbers: ", num1 / num2)
else:
    print("Invalid option.")

"Question 3"
inpt_sen = input("Enter the sentence you want to manipulate: ")
print(inpt_sen.replace("python", "pythons"))

"Bonus"
"1. armstrong number "
num = int(input("Enter a number: "))
sum = 0
temp = num
while temp > 0:
    digit = temp % 10
    sum += digit ** 3
    temp //= 10

if num == sum:
    print(num,"is an Armstrong number")
else:
    print(num,"is not an Armstrong number")

"2. Palindrome string or not"
pali_stn = input("Enter palindrome string: ")
rev_pali_str = pali_stn[::-1]
if pali_stn == rev_pali_str:
    print("String is Palindrome.")
else:
    print("String is not Palindrome.")
