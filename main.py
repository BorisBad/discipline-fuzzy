import pandas as pd
import numpy as np
import itertools
from timeit import default_timer as timer
from multiprocessing import Pool
#import os
#import datetime

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
            end = crash_point
    return [start, end, gs[0], gs[1]]

def compute_g(list_of_timestamps:list, normal_hist:list, n_edges:list, crash_hist:list, c_edges:list):
    normal_sum = 0
    crash_sum = 0
    mx_n_e = max(n_edges)
    mn_n_e = min(n_edges)
    l_n_e = len(n_edges)
    l_n_h = len(normal_hist)
    mx_c_e = max(c_edges)
    mn_c_e = min(c_edges)
    l_c_e = len(c_edges)
    l_c_h = len(crash_hist)        
    for i in range(0,len(list_of_timestamps)):
        if mn_n_e <= list_of_timestamps[i] <= mx_n_e:
            step = (mx_n_e-mn_n_e)/(l_n_e-1)
            idx = int(list_of_timestamps[i]//(step+mn_n_e))
            if idx == l_n_h:
                normal_sum += normal_hist[idx-1]
            else:
                normal_sum += normal_hist[idx]
        if mn_c_e <= list_of_timestamps[i] <= mx_c_e:
            step = (mx_c_e-mn_c_e)/(l_c_e-1)
            idx = int(list_of_timestamps[i]//(step+mn_c_e))
            if idx == l_c_h:
                normal_sum += crash_hist[idx-1]
            else:
                normal_sum += crash_hist[idx]  
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

    #step = datetime.datetime.now()
    #tm = timer()
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
    #print(f"id{os.getpid()}:time{datetime.datetime.now()-step}, {n,m,k}")
    #print(f"id{os.getpid()}:time{timer()-tm}, {n,m,k}")
    return answer

def collect_results(answer:pd.DataFrame, tasks:list):
    rez = []
    if __name__ == '__main__':
        with Pool(glob_workers) as pool_processes:
            rez = pool_processes.starmap(find_answer,tasks)
    for i in range(0, len(rez)):
        answer = pd.concat([answer,rez[i]])
        #print('i {}: {}'.format(i, len(answer)))
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
    for n in range(1000, 7000, 1000):
        for m, k in itertools.product(range(n//3, n//2, 100), range(1000, n, 1000)):
            tasks.append((n,m,k))
    
    if __name__ =='__main__':
        tmg = timer()
        print(f"Startesd. approximate time is {len(tasks)*0.5/60/60} - {len(tasks)*0.5/60/60/glob_workers} hours")
    answer = collect_results(answer, tasks)
    if __name__ =='__main__':
        print('Done')
        print(f'Time taken: {(timer()-tmg)/60/60} hours')
        answer.to_csv(f'answer - {len(tasks)}.csv', encoding='utf-8', sep=';', header=True)
    return

main()