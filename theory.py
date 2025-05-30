from math import exp,log


def multistep_calc(x):
    return 0.15*(6*exp(-2*(4*x+2)**(-1/17)*log(1/(4*x+2)))+3*3.556678+3*(10+x)*(4*x+2)**2)

def cot_tilde_1(x):
    return 0.3*exp(-2*(31.24*(x/(1+x)))**(-1/13)*log(1/(31.24*(x/(1+x))))) #, x=0.6667

def cot_tilde_2(x):
    return 0.3*exp(-2*(31.24*(x/(1+x)))**(-1/25)*log(1/(31.24*(x/(1+x))))) #, x=0.6667

def cot_bar_1(x):
    return 0.3*exp(-2*(15.62*x)**(-1/13)*log(1/(15.62*x)))

def cot_bar_2(x):
    return 0.3*exp(-2*(15.62*x)**(-1/25)*log(1/(15.62*x)))


print("--multistep_calc--")
for i in [0,1,2,3,4,5]:
    print("%s %.5f \\\\"%(i,multistep_calc(i)))

print("\n")
print("--tilde_1--")
for i in [0.0526, 0.1111, 0.25, 0.4286, 0.6667, 1.0]:
    print("%s %.5f \\\\" % (i, cot_tilde_1(i)))

print("\n")
print("--tilde_2--")
for i in [0.0526, 0.1111, 0.25, 0.4286, 0.6667, 1.0]:
    print("%s %.5f \\\\" % (i, cot_tilde_2(i)))

print("\n")
print("--bar_1--")
for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    print("%s %.5f \\\\" % (i, cot_bar_1(i)))

print("\n")
print("--bar_2--")
for i in [0.05,0.1,0.2,0.3,0.4,0.5]:
    print("%s %.5f \\\\" % (i, cot_bar_2(i)))
