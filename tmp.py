import matplotlib.pyplot as plt

# First plot
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.title('First Plot')
plt.show(block=False)

# Second plot
plt.figure()
plt.plot([1, 2, 3], [6, 5, 4])
plt.title('Second Plot')
plt.show(block=False)

# Third plot
plt.figure()
plt.plot([1, 2, 3], [7, 8, 9])
plt.title('Third Plot')
plt.show(block=False)

# Keep the plots open
input("Press Enter to close all plots...")
