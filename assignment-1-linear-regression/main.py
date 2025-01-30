import sys
import pandas as pd

def main(argv):
    # read csv file into a DataFrame
    filename = "data.csv"
    data = pd.read_csv(filename, dtype={"Cabin": str})

    # first few rows of the DataFrame
    print(data.head())
    

if __name__ == "__main__":
    main(sys.argv)