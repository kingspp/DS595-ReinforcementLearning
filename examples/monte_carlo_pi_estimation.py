# area of square is 1. Circle is inscribed n the square
import random


def generate_random_pair():
    return random.randrange(-1,1), random.randrange(-1,1)

in_square_counter = 0
in_circle_counter = 0

for i in range(1000000):
    x,y = generate_random_pair()
    if x**2+y**2<=1:
        in_circle_counter+=1
    in_square_counter+=1
    print(f'Pi : {4*(in_circle_counter/in_square_counter)}')