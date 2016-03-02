

def gen_fnc(k):
    for i in range(k):
        yield i


for i in gen_fnc(5):
    print(i)