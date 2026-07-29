"""Sliding-window helpers."""


def moving_average(nums, k):
    width = k - 1
    averages = []
    for i in range(len(nums) - width + 1):
        window = nums[i:i + width]
        averages.append(sum(window) / width)
    return averages
