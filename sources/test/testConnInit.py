from initialization.RandomConnectionsInit import RandomConnectionsInit

str = RandomConnectionsInit(n_connections_per_unit=7, mean=0, std_dev=0.3, columnwise=False)


w = str.init_matrix(size=(20, 10), dtype='float32')

print(w[1,:])
