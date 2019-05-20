class Rectangle:
    def __init__(self, lower_left=(0,0), height=0, width=0):
        self.lower_left = lower_left
        self.height = height
        self.width = width

class Circle:
    def __init__(self, center=(0,0), radius=0):
        self.center = center
        self.radius = radius

class Runner: 
    def __init__(self, lower_left=(0,0), height=0, width=0):
        self.lower_left = lower_left
        self.height = height
        self.width = width