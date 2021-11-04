import matplotlib.pyplot as plt


def batch_image_peak(batch_x, batch_y):
    fig, m_axs = plt.subplots(4, 4, figsize=(16, 16))
    for (c_x, c_y, c_ax) in zip(batch_x, batch_y, m_axs.flatten()):
        c_ax.imshow(c_x[:, :, 0], cmap='bone')
        if c_y == 1:
            c_ax.set_title('Pneumonia')
        else:
            c_ax.set_title('No Pneumonia')
        c_ax.axis('off')
    plt.show()


def batch_histogram_peak(batch_x, batch_y):
    fig, m_axs = plt.subplots(4, 4, figsize=(16, 16))
    for (c_x, c_y, c_ax) in zip(batch_x, batch_y, m_axs.flatten()):
        c_ax.hist(c_x[:, :, 0].ravel(), bins=256)
        if c_y == 1:
            c_ax.set_title('Pneumonia')
        else:
            c_ax.set_title('No Pneumonia')
    plt.show()
