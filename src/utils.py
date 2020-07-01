import h5py
import datetime
from config import START_DATE, DATE_FORMAT, START_DATETIME

def read_depth(data, x, z):
    """ Read data from a simple depth file (x,z) along a set y coordinate corresponding to a circle

    Args:
        filepath: path of file
        x: x value
        z: depth

    Returns:
        value for each (x,z) coordinate

    """

    # data = pandas.read_csv(filepath) # put in main?
    value = data[str(int(z))][int(x)]
    
    return value

def read_stock_history(filepath):
    """ Read data from extracted h5

    Args:
        filepath: path of file

    Returns:
        history:
        abbreviation:

    """
    with h5py.File(filepath, 'r') as f:
        history = f['history'][:]
        abbreviation = f['abbreviation'][:].tolist()
        abbreviation = [abbr.decode('utf-8') for abbr in abbreviation]
    return history, abbreviation


def index_to_date(index):
    return (START_DATETIME + datetime.timedelta(index)).strftime(DATE_FORMAT)


def date_to_index(date_string):
    return (datetime.datetime.strptime(date_string, DATE_FORMAT) - START_DATETIME).days