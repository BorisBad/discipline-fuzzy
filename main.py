import pandas as pd
import numpy as np
import itertools
from timeit import default_timer as timer

#read from csv
def read_from_file(file_name:str):
    return pd.read_csv(file_name, sep=';')

def find_stop_point(df:pd.DataFrame, n_h:np.ndarray, n_e:np.ndarray, c_h:np.ndarray, c_e:np.ndarray, k:int):
    start = 0
    end = k
    gl = [0, 0]
    crash_point = df['Values'].idxmax()
    while end <= crash_point:
        gs = compute_g(df['Values'].iloc[start:end].to_list(),n_h,n_e,c_h,c_e)
        if gs[1] > gs[0]:
            break
        if gs != gl:
            gl = gs    
        start = end+1
        end += k
        if end > len(df['Values']):
            end = df['Values'].iloc[-1]
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

def main():
    #1. read crashes from  csv files
    #region
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
    #endregion

    #optimal values
    # n = 3500
    # m = n//2
    # k = 500

    #2. 
    #create df for answer
    #region
    answer = pd.DataFrame(columns=['n','m','k','g1_1','g2_1','start_1','end_1','g1_2','g2_2','start_2','end_2'])
    #endregion

    #iterate over n,m,k to find optimal answer (brute force)
    #region
    curstep = 0
    for n in range(500, 50000, 500):
        for m, k in itertools.product(range(2, n//2, 1), range(100, n, 100)):
            tm = timer()
            #3.1 First N histogram (normalwork)
            #region
            normal_work_histogram = np.histogram(crash_1['Values'].iloc[0:n],bins=m)
            normal_work_hist = normal_work_histogram[0]/sum(normal_work_histogram[0])
            normal_work_edges = normal_work_histogram[1]
            #endregion

            #3.2 Last N histogram (N timestamps before crash wich is max value)
            #region
            crash_point = crash_1['Values'].idxmax()
            crash_work_histogram = np.histogram(crash_1['Values'].iloc[crash_point-n:crash_point], m)
            crash_work_hist = crash_work_histogram[0]/sum(crash_work_histogram[0])
            crash_work_edges = crash_work_histogram[1]
            #endregion
            
            #4. Compute g1(x) and g2(x) for K timestamps for each file
            #region
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
            print('step {} is done, time taken: {} (sec)'.format(curstep,timer()-tm))
            curstep += 1
        #endregion
    #endregion

    #5. Print answer
    #region
    answer.to_csv('answer.csv', encoding='utf-8', sep=';', header=True)
    #endregion
    
    return

main()