import numpy as np

mat = np.zeros((6, 6), dtype=int)

rows_to_fill = 2
index = 0
count = 0   # how many rows we’ve processed since skipping

for i in range(rows_to_fill, mat.shape[0]):   # skip the first 2 rows
    #for j in range(mat.shape[1]):      # cols
    mat[i][index] += 1
    count += 1
    if count % rows_to_fill == 0:  # every rows_to_fill rows
        index += 1
print(mat)
np.save("./custom.npy", mat)
print("custom matrix saved!")