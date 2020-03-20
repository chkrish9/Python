"Question 1"
totalwgts = int(input("Enter the number students: "))
lbsLst = []
kgLst = []
"Without List Comprehensions"
"""
for i in range(totalwgts):
    if i == 0:
        wgt = float(input(f'Enter {i+1}st student weight in lbs:'))
    elif i == 1:
        wgt = float(input(f'Enter {i + 1}nd student weight in lbs:'))
    elif i == 2:
        wgt = float(input(f'Enter {i + 1}rd student weight in lbs:'))
    else:
        wgt = float(input(f'Enter {i + 1}th student weight in lbs:'))
    lbsLst.append(wgt)
    kgLst.append(wgt * 0.453592)
"""
"With List Comprehensions"
lbsLst = [float(input(f'Enter {i + 1} student weight in lbs:')) for i in range(totalwgts)]
kgLst = [lbsLst[i] * 0.453592 for i in range(len(lbsLst))]
print(f"Student weights in lbs:{lbsLst}")
print(f"Student weights in kgs:{kgLst}")


