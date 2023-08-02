import pandas as pd
import os 
def read_data():
    # Change directory to ./output
    os.chdir('./output')

    # Read data

    b1 = pd.read_csv('B1_20230801-112958.csv')
    b2 = pd.read_csv('B2_20230801-113343.csv')
    b3 = pd.read_csv('B3_20230801-113729.csv')
    b4 = pd.read_csv('B4_20230801-114117.csv')
    b5 = pd.read_csv('B5_20230801-114506.csv')

    c1 = pd.read_csv('C1_20230801-114852.csv')
    c2 = pd.read_csv('C2_20230801-115321.csv')
    c3 = pd.read_csv('C3_20230801-115750.csv')
    c4 = pd.read_csv('C4_20230801-120220.csv')
    c5 = pd.read_csv('C5_20230801-120653.csv')

    d1 = pd.read_csv('D1_20230801-121158.csv')
    d2 = pd.read_csv('D2_20230801-121704.csv')
    d3 = pd.read_csv('D3_20230801-122212.csv')
    d4 = pd.read_csv('D4_20230801-122723.csv')
    d5 = pd.read_csv('D5_20230801-123242.csv')

    e1 = pd.read_csv('E1_20230801-123829.csv')
    e2 = pd.read_csv('E2_20230801-124421.csv')
    e3 = pd.read_csv('E3_20230801-125009.csv')
    e4 = pd.read_csv('E4_20230801-125606.csv')
    e5 = pd.read_csv('E5_20230801-130210.csv')

    # Use a for loop to iterate through all files
    # extract the row of 'val_loss'
    # and for that row, read the 'test_acc'
    # then store it into a list
    # Repeat for all files
    lst = []
    for i in range(1, 6):
        b = eval('b' + str(i))
        c = eval('c' + str(i))
        d = eval('d' + str(i))
        e = eval('e' + str(i))

        b_val_loss = b['valid_loss']
        b_test_acc = b.loc[b_val_loss.idxmin(), 'test_auc']

        c_val_loss = c['valid_loss']
        c_test_acc = c.loc[c_val_loss.idxmin(), 'test_auc']

        d_val_loss = d['valid_loss']
        d_test_acc = d.loc[d_val_loss.idxmin(), 'test_auc']

        e_val_loss = e['valid_loss']
        e_test_acc = e.loc[e_val_loss.idxmin(), 'test_auc']

        lst.append([b_test_acc, c_test_acc, d_test_acc, e_test_acc])

        # lst.append([b_test_acc, c_test_acc])

    # Convert list to dataframe
    df = pd.DataFrame(lst, columns=['B', 'C', 'D', 'E'])

    # output as B1_E5_summary.csv
    df.to_csv('B1_E5_summary.csv', index=False)

if __name__ == "__main__":
    read_data()

    