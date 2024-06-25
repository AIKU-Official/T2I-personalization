from itertools import product

def get_total_step_layer():
    total_pairs = \
        [(4, i, 1, list(range(4, i+1))) for i in range(20, 51, 5)] + \
        [(i, 50, 1, list(range(i, 51))) for i in range(0, 11, 2)] + \
        [(4, 50, i, [j for j in range(4, 51) if j % 10 >= i]) for i in range(1, 10, 1)]
    
    layer_pairs = [[(10,11,12,13,14,15),"Decoder"], [(0,1,2,3,4,5,6,7),"Encoder"]]
    print(total_pairs)
    return total_pairs, layer_pairs

