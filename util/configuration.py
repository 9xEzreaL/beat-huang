dataset1 = '/media/ExtHDD01/Dataset/WBC/WBC_dataset2'
dataset2 = '/media/ExtHDD01/Dataset/WBC/WBC_dataset3'
dataset3 = '/media/ExtHDD01/Dataset/WBC/WBC_dataset4'

blood_type = ['blast', 'promyelo', 'myelo', 'meta', 'band', 'seg']

total_num = {
    'dataset1': {'blast': 0, 'promyelo': 565, 'myelo':1110 ,'meta':962 ,'band':1891 , 'seg':1510},
    'dataset2': {'blast': 2962, 'promyelo': 512, 'myelo':992 ,'meta':658 ,'band':2095 , 'seg':2215},
    'dataset3': {'blast': 5279, 'promyelo': 352, 'myelo':339 ,'meta':215 ,'band':815 , 'seg':2313}
}

split = {
        'train': {'blast': {'dataset1': None, 'dataset2': [0, 2632], 'dataset3': [0, 4693]},
                  'promyelo': {'dataset1': [0, 427], 'dataset2': [0, 427], 'dataset3': [0, 284]},
                  'myelo': {'dataset1': [0, 955], 'dataset2': [0, 854], 'dataset3': [0, 283]},
                  'meta': {'dataset1': [0, 815], 'dataset2': [0, 557], 'dataset3': [0, 180]},
                  'band': {'dataset1': [0, 1670], 'dataset2': [0, 1837], 'dataset3': [0, 709]},
                  'seg': {'dataset1': [0, 1335], 'dataset2': [0, 1958], 'dataset3': [0, 2027]}},

         'eval': {'blast': {'dataset1': None, 'dataset2': [2632, 2632+34], 'dataset3': [4693, 4693+66]},
                  'promyelo': {'dataset1': [427, 427+37], 'dataset2': [427, 427+34], 'dataset3': [284, 284+29]},
                  'myelo': {'dataset1': [955, 955+44], 'dataset2': [854, 854+39], 'dataset3': [283, 283+17]},
                  'meta': {'dataset1': [815, 815+51], 'dataset2': [557, 557+35], 'dataset3': [180, 180+14]},
                  'band': {'dataset1': [1670, 1670+32], 'dataset2': [1837, 1837+48], 'dataset3': [709, 709+20]},
                  'seg': {'dataset1': [1335, 1335+24], 'dataset2': [1958, 1958+35], 'dataset3': [2027, 2027+41]}},

         'test': {'blast': {'dataset1': None, 'dataset2': [2632+34, 2632+34+296], 'dataset3': [4693+66, 4693+66+520]},
                  'promyelo': {'dataset1': [427+37, 427+37+57], 'dataset2': [427+34, 427+34+51], 'dataset3': [284+29, 284+29+39]},
                  'myelo': {'dataset1': [955+44, 955+44+111], 'dataset2': [854+39, 854+39+99], 'dataset3': [283+17, 283+17+39]},
                  'meta': {'dataset1': [815+51, 815+51+96], 'dataset2': [557+35, 557+35+66], 'dataset3': [180+14, 180+14+21]},
                  'band': {'dataset1': [1670+32, 1670+32+189], 'dataset2': [1837+48, 1837+48+210], 'dataset3': [709+20, 709+20+86]},
                  'seg': {'dataset1': [1335+24, 1335+24+151], 'dataset2': [1958+35, 1958+35+222], 'dataset3': [2027+41, 2027+41+245]}}
         }

save_logs = '/media/ExtHDD01/wbc_logs'