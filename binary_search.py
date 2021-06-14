import numpy as np


# Python3 program to find element
# closet to given target.

# Returns element closest to target in arr[]
def findClosest(arr, n, target):
    # Corner cases
    if (target <= arr[0][0]):
        return 0
    if (target >= arr[n - 1][0]):
        return n - 1

    # Doing binary search
    i = 0
    j = n
    mid = 0
    while (i < j):
        mid = (i + j) // 2

        if (arr[mid][0] == target):
            return mid

        # If target is less than array
        # element, then search in left
        if (target < arr[mid][0]):

            # If target is greater than previous
            # to mid, return closest of two
            if (mid > 0 and target > arr[mid - 1][0]):
                return getClosest(arr, mid - 1, mid, target)

            # Repeat for left half
            j = mid

        # If target is greater than mid
        else:
            if (mid < n - 1 and target < arr[mid + 1][0]):
                return getClosest(arr, mid, mid + 1, target)

            # update i
            i = mid + 1

    # Only single element left after search
    return mid


# Method to compare which one is the more close.
# We find the closest by taking the difference
# between the target and both values. It assumes
# that val2 is greater than val1 and target lies
# between these two.
def getClosest(arr, ind1, ind2, target):
    val1 = arr[ind1][0]
    val2 = arr[ind2][0]
    if (target - val1 >= val2 - target):
        return ind2
    else:
        return ind1


def get_bound(arr, N, s,e):

    f1 = get_bound_util(arr, N, s, True)
    f2 = get_bound_util(arr, N, e, False)
    return f1,f2


def get_bound_util(arr, N, X, is_start):

    if is_start:
        idx = findClosest(arr, N, X)
        # idx = 0
        if idx==0:
            return np.zeros(60)
        else:
            return arr[idx-1][1:]
    else:
        idx = findClosest(arr, arr.shape[0], X)
        # idx = N-1
        return arr[idx][1:]

if __name__ == '__main__':
    gb = get_bound([[4], [5], [10], [12], [18], [20]], 6, 20, True)
    print(gb)
