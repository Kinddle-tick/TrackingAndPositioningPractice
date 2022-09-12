# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from algorithm_3 import *
from showmaker import *

def f(a,b,c):
    ba = b-a
    ca = c-a
    if a > 1:
        return 2*(ba+ca)+min(b,c)+(a-1)
    else:
        return 2*(ba+ca)+min(b,c)+(a-1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    length = 7
    width = 7

    #square
    sum_sq = 0
    for tmp_l in range(1,length+1):
        for tmp_w in range(1,width+1):
            add_tmp = (length-tmp_l+1)*(width-tmp_w+1)
            sum_sq += add_tmp
            if tmp_w == tmp_l:
                sum_sq += 2 * (tmp_w-1) *add_tmp
            else:
                sum_sq += 2 * add_tmp
            # if tmp_w == tmp_l:
            #     sum_sq += add_tmp
            # else:
            #     sum_sq += add_tmp
            print((length-tmp_l+1), (width-tmp_w+1))
    print(sum_sq)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
