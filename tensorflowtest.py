import tensorflow as tf

# List all physical devices
devices = tf.config.list_physical_devices()
print("Available devices:", devices)

# List GPUs specifically
gpu_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpu_devices)

# List CPUs specifically
cpu_devices = tf.config.list_physical_devices('CPU')
print("Available CPUs:", cpu_devices)