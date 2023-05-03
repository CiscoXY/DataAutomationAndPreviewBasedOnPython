from multiprocessing import Pool
from threading import Thread
import time
import numpy as np
def Pool_test_func(matrixA , matrixB):
    
    return matrixA @ matrixB

np.random.seed(3000)

args1 = [np.random.randn(3000 , 3000) , np.random.randn(3000 , 3000)]
args2 = [np.random.randn(3000 , 3000) , np.random.randn(3000 , 3000)]
args3 = [np.random.randn(3000 , 3000) , np.random.randn(3000 , 3000)]
args4 = [np.random.randn(3000 , 3000) , np.random.randn(3000 , 3000)]

args_list = [args1, args2, args3, args4]



# def Pool_test_func(a , b , c , d = ' fuck' , e = 'what? ' , f = '.end'):
#     return a + b + c + d , e+f

# args1 = ('a' , 'b' , 'c' , 'd')
# args2 = ['a' , 'b' , 'c']
# args3 = ['args3 a' , 'args3 b' , 'args 3 c' , 'final']
# args4 = ['args3 a ' , 'args3 b ' , 'args4 c ' , 'final' , '' , '']
# args_list = [args1, args2, args3, args4]  # 4 个参数组


if __name__ == '__main__':
    tic = time.time()
    for i in args_list:
        Pool_test_func(i[0] , i[1])
    tac = time.time()
    print(f'time use: {tac - tic}s')
    
    tic = time.time()
    with Pool(4) as p:  # 使用 4 个进程
        results = p.starmap(Pool_test_func, args_list)
    # t1 = Thread(target=Pool_test_func, args = args_list[0])
    # t1.start()
    # t2 = Thread(target=Pool_test_func, args = args_list[1])
    # t2.start()
    # t3 = Thread(target=Pool_test_func, args = args_list[2])
    # t3.start()
    # t4 = Thread(target=Pool_test_func, args = args_list[3])
    # t4.start()
    
    # t1.join()
    # t2.join()
    # t3.join()
    # t4.join()
    
    tac = time.time()
    print(f'time use: {tac - tic}s')
    print(results)