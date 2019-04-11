



class Function_Class():

    def print_success(self):
        print("successfull")

    def add(self, x, y):
        self.print_success()
        return x + y 

    def sub(self, x, y):
        self.print_success()
        return x - y 

class Execute_Class():
    def __init__(self, function):
        self.function = function
    
    def execute(self, x, y): 
        return self.function(x, y) 



x = 10 
y = 2 

f = Function_Class()
ex = Execute_Class(f.add)


result =  ex.execute(10, 2)

print("Result:", result )