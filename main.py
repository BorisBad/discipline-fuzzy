import pandas as pd
import numpy as np
import itertools
from timeit import default_timer as timer
from multiprocessing import Pool
import os
import datetime

glob_workers = 4

#read from csv
def read_from_file(file_name:str):
    return pd.read_csv(file_name, sep=';')

def find_stop_point(df:pd.DataFrame, n_h:np.ndarray, n_e:np.ndarray, c_h:np.ndarray, c_e:np.ndarray, k:int):
    start = 0
    end = k
    gl = [0, 0]

    crash_point = len(df['Values']-1)
    while end < crash_point:
        gs = compute_g(df['Values'].iloc[start:end].to_list(),n_h,n_e,c_h,c_e)
        if gs[1] > gs[0]:
            break
        if gs != gl:
            gl = gs    
        start = end+1
        end += k
        if (end > crash_point) and (start < crash_point):
            end = len(df['Values'])
    return [start, end, gs[0], gs[1]]

def compute_g(list_of_timestamps:list, normal_hist:list, n_edges:list, crash_hist:list, c_edges:list):
    normal_sum = 0
    crash_sum = 0
    for i in range(0,len(list_of_timestamps)):
        if n_edges[0] <= list_of_timestamps[i] <= n_edges[-1]:
            step = (n_edges[-1]-n_edges[0])/len(n_edges)
            normal_sum += normal_hist[int(list_of_timestamps[i]//(step+n_edges[0]))]
        if c_edges[0] <= list_of_timestamps[i] <= c_edges[-1]:
            step = (c_edges[-1]-c_edges[0])/len(c_edges)
            crash_sum += crash_hist[int(list_of_timestamps[i]//(step+c_edges[0]))]    
    return [normal_sum, crash_sum]

def sliding_window(items, size):
    return [items[start:end] for start, end
            in zip(range(0, len(items) - size + 1), range(size, len(items) + 1))]

def find_answer(n:int,m:int,k:int):
    try:
        crash_1 = read_from_file('1.csv')
        crash_2 = read_from_file('2.csv')
    except OSError as err:
        print("Oops OSError", err)
        return
    except ValueError as err:
        print("Oops ValueError", err)
        return
    except Exception as err:
        print("Oops Exception", err)
        return

    step = datetime.datetime.now()
    answer = pd.DataFrame(columns=['n','m','k','g1_1','g2_1','start_1','end_1','g1_2','g2_2','start_2','end_2'])
    normal_work = np.histogram(crash_1['Values'].iloc[0:n],bins=m)
    normal_work_hist = normal_work[0]/sum(normal_work[0])
    normal_work_edges = normal_work[1]
    crash_point = len(crash_1['Values']-1)
    crash_work = np.histogram(crash_1['Values'].iloc[crash_point-n:crash_point], m)
    crash_work_hist = crash_work[0]/sum(crash_work[0])
    crash_work_edges = crash_work[1]
    res1 = find_stop_point(crash_1,normal_work_hist,normal_work_edges,crash_work_hist,crash_work_edges,k)
    res2 = find_stop_point(crash_2,normal_work_hist,normal_work_edges,crash_work_hist,crash_work_edges,k)
    answer = pd.concat(
                    [pd.DataFrame([[n,
                                    m,
                                    k,
                                    res1[2],
                                    res1[3],
                                    res1[0],
                                    res1[1],
                                    res2[2],
                                    res2[3],
                                    res2[0],
                                    res2[1]]], columns=answer.columns), answer],
                    ignore_index=True
                )
    print(f"id{os.getpid()}:time{datetime.datetime.now()-step}, {n,m,k}")
    return answer

def collect_results(answer:pd.DataFrame, tasks:list):
    rez = []
    if __name__ == '__main__':
        with Pool(glob_workers) as pool_processes:
            rez = pool_processes.starmap(find_answer,tasks)
    for i in range(0, len(rez)):
        answer = pd.concat([answer,rez[i]])
        print('i {}: {}'.format(i, len(answer)))
    return answer

def main():
    #optimal values
    # n = 3500
    # m = n//2
    # k = 500

    #create df for answer
    answer = pd.DataFrame(columns=['n','m','k','g1_1','g2_1','start_1','end_1','g1_2','g2_2','start_2','end_2'])

    #iterate over n,m,k to find optimal answer (brute force)
    tasks = []
    for n in range(500, 2000, 500):
        for m, k in itertools.product(range(100, n//2, 100), range(100, n, 100)):
            tasks.append((n,m,k))
    
    answer = collect_results(answer, tasks)
    answer.to_csv('answer.csv', encoding='utf-8', sep=';', header=True)
    return

main()