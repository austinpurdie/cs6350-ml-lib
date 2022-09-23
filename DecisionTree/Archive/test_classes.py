import pandas as pd

dict = {'beep': [[1, 2], [3, 4]], 
        'boop': [[5, 6], [7, 8]],
        'bop': [[9, 10], [11, 12]]
}

df = pd.DataFrame.from_dict(dict)

print(df)