"Question 1"


class Employee:
    empcount = 0
    avgsal = 0

    def __init__(self, fname, lname, salary, dept):
        self.fname = fname
        self.lname = lname
        self.salary = salary
        self.dept = dept
        Employee.empcount += 1
        Employee.avgsal += salary

    def averagesal(self):
        print("The average salary of the employees:", Employee.avgsal/Employee.empcount)


class FullTimeEmployee(Employee):
    def displayEmplyee(self):
        print("Full name of the employee:", self.fname, self.lname)


fEmployee = FullTimeEmployee("Murali", "C", 1000, "CS")
fEmployee.displayEmplyee()
fEmployee.averagesal()

fEmployee = FullTimeEmployee("Krishna", "C", 2000, "CS")
fEmployee.displayEmplyee()
fEmployee.averagesal()