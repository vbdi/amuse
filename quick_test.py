def factorial( n):
    v = 1
    while n > 0:
        v *= n
        n = n - 1
    return v

def choose(n, r):

    return factorial(n) // (factorial(r) * factorial(n - r))

def main():
    counter = 0
    counter_total = 0
    for n in range(1, 100 + 1):
        for k in range(1, n):
            counter_total += 1
            if k / n > 0.60:
                counter +=1
                continue

            choose(n, n - k)
    print(counter, counter_total)

if __name__=='__main__':
    main()